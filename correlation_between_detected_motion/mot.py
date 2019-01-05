import cv2

import multiprocessing as mp
import string
import numpy as np
import time#To calculate how long it takes to compute n frames
import sys
import os
import tables
import base64#To write log file
import csv
import datetime

try:
    import pyflow
except:
    print("Brocks dense optical flow not available")

class ComputeMotion():
    def __init__(self, filename, outpath= None, algorithm = "farneback", supress_output = False, batch_size = 8,
                scaled_size = (320,180), output_scaling = 0.5, use_multiprocessing = True):

        self.use_multiprocessing = use_multiprocessing
        #For file input/output
        self.filename = filename
        self.outpath = outpath
        self.save_chunksize = 100

        #Essential for processing
        self.algorithm = algorithm
        self.cap = cv2.VideoCapture(filename)
        self.scaled_size = scaled_size#ATTENTION: FOR OPENCV WIDTH FIRTST AND NOT ALL SIZES WORK WITH farneback
        self.start_pos = 0 #current position
        self.batch_size = batch_size
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.buffer = []
        self.interpolation = cv2.INTER_CUBIC#Fastest for resize
        self.previous_frame = None #Will temporarily hold the last frame that was loaded by self.get_two_frames()

        #Maily for logging and communicating
        self.supress_output = supress_output
        self.total_time_estimate = 0
        self.percent_completed = 0
        self.n_pairs_to_read = None
        self.do_write_logfile = True
        self.additional_logmessage= ""
        self.absolute_start_pos = None #The one used by save to file

        self.set_output_dimensions(output_scaling)

    def farneback_wrapper(self, pos, sample, output = None):
        out = cv2.calcOpticalFlowFarneback(sample[0], sample[1], None, 0.5, 3, 15, 2, 5, 1.2, 0)
        if output:
            output.put((pos, out))
        else:
            return out

    def write_logfile(self, frame):
        logfile = self.outpath + ".log.csv"
        now = str(datetime.datetime.now())
        if(os.path.isfile(logfile)):#File exists --> Append
            with open(logfile, 'a', encoding='utf8') as csv_file:
                wr = csv.writer(csv_file, delimiter='\t')
                wr.writerow([now, frame, str(self.total_time_estimate), str(self.percent_completed)+"%",self.additional_logmessage])
                self.additional_logmessage = ""
        else:
            with open(logfile, 'a', encoding='utf8') as csv_file:
                wr = csv.writer(csv_file, delimiter='\t')
                wr.writerow(['Current date', 'Current frame', 'Total time estimate',' % Completed','Additional Message'])

    def set_output_dimensions(self, output_scaling):#Has sideeffect of resetting framenumber do only call before processing starts
        #Get result for first pair of data, create file with appropriate dimensions and write to it
        q = mp.Queue()
        frames_left, pair_1 = self.get_two_frames()

        if(self.algorithm == "farneback"):
            self.farneback_wrapper(0,pair_1,q)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,0)#reset framenumber

        if(self.algorithm == "brocks"):
            self.brocks_wrapper(0,pair_1,q)
        first_result = np.array(q.get()[1])
        self.output_dim = first_result.shape
        self.outfile_shape = [0]
        self.outfile_shape.extend(list(self.output_dim))#HDF5 via tables requires a 0 in first dimension

        self.outfile_shape[1] *= output_scaling
        self.outfile_shape[2] *= output_scaling

        self.outfile_shape = np.array(self.outfile_shape, dtype = np.int)
        assert self.outfile_shape[0] == 0

    def write_hdf5(self, retry_writing = 10):
        print("write to file")
        successfully_written = False
        if(self.buffer.shape[0]==0):#If buffer is empty dont try to write
            return -1

        shape = np.copy(self.outfile_shape) #i.e. [0,width,height,2]
        shape[0] = self.buffer.shape[0] #e.g. [100,width,height,2]
        rescaled = np.ndarray(shape = shape)
        for result, index in zip(self.buffer, range(rescaled.shape[0])):
            rescaled[index,:,:,0] = cv2.resize(self.buffer[index,:,:,0], dsize=tuple([shape[2], shape[1]]))
            rescaled[index,:,:,1] = cv2.resize(self.buffer[index,:,:,1], dsize=tuple([shape[2], shape[1]]))


        for x in range(10):
            if(os.path.isfile(self.outpath)):#File exists --> Append
                try:
                    f = tables.open_file(self.outpath, mode='a')
                    f.root.motion_tensor.append(rescaled)
                    self.buffer = []#Empty output queue
                    successfully_written = True
                    f.close()
                except Exception as e:
                    self.additional_logmessage += "Writing to file was not possible. "


            else:
                try:
                    f = tables.open_file(self.outpath, mode='w')
                    array_c = f.create_earray(f.root, 'motion_tensor', tables.Float32Atom(), self.outfile_shape)
                    array_c.append(rescaled) # Writing requires [[DATA]] i.e.shape (0,..)
                    self.buffer = []
                    successfully_written = True
                    f.close()
                except Exception as e:
                    print("Writing to file was not possible")
                    self.additional_logmessage += "Writing to file was not possible. "
            if(successfully_written):
                print("sucessfully written")
                break
            else:
                self.additional_logmessage += "retry writing"
                print("retry writing")
        return successfully_written


    def brocks_wrapper(self, pos, sample, output = None):
        im1 = sample[0]
        im2 = sample[1]
        im1 = im1.astype(float) / 255.
        im2 = im2.astype(float) / 255.
        im1 = np.stack((im1,), -1)
        im2 = np.stack((im2,), -1)

        # Flow Options:
        alpha = 0.012
        ratio = 0.75
        minWidth = 5
        nOuterFPIterations = 1
        nInnerFPIterations = 1
        nSORIterations = 1
        colType = 1  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

        u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations, colType)
        flow = np.concatenate((u[..., None], v[..., None]), axis=2)

        if output:
            output.put([pos, flow])
        else:
            return flow

    def get_two_frames(self):
        cap = self.cap

        ret = False
        if(type(self.previous_frame)==type(None)):#If there is no previous frame load one and scale it
            ret, frame1 = cap.read()
            if(not ret):
                self.additional_logmessage += "cap returned FALSE; end of file"
                return (False, None)
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)#
            frame1 = cv2.resize(frame1, dsize=self.scaled_size, interpolation=self.interpolation)
            self.previous_frame = frame1 #self.previous_frame is always already rescaled

        ret, frame2 = cap.read()
        if(not ret):
            self.additional_logmessage += "cap returned FALSE; end of file"
            return (False, None)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.resize(frame2, dsize=self.scaled_size, interpolation=self.interpolation)

        frame1 = self.previous_frame
        self.previous_frame = frame2
        assert frame1.shape == frame2.shape

        return (True, (frame1, frame2))

    def get_batches_of_pairs(self, batch_size):
        ret = []
        for pair in range(batch_size):
            data_left, frames = self.get_two_frames()
            if(not data_left):
                return (False, ret)
            else:
                ret.append(frames)
        return (True, ret)#(last pair?, data)

    def to_file(self, start_pos = 0, n_pairs_to_read = None):
        self.cap.set(1,start_pos)
        self.absolute_start_pos = start_pos

        if(self.outpath == None):
            print("Set outputpath first")
            return -1

        if(n_pairs_to_read==None):#Assume having to read all frames if no parameter is set
            n_pairs_to_read = self.total_frames
            if(n_pairs_to_read <= 0):#Assume having to read all frames if no parameter is set
                print("The obtained filesize was 0")
                print("Corrupted files might be processed either way by passing 0 for start pos and a value for n_pairs_to_read")
                return -1


        self.n_pairs_to_read = n_pairs_to_read#Make available for print method

        cycles = n_pairs_to_read//self.save_chunksize
        rest = n_pairs_to_read%self.save_chunksize


        for x in range(cycles):
            self.get_motion(self.save_chunksize)#fill buffer

            if(self.buffer.shape[0]== 0):# If file is corrupted n_pairs_to_read might be larger then it actually is
                break # Therefore it's necessary to check wheather the buffer was actually filled

            self.write_hdf5()#write to file

            if(self.do_write_logfile == True):
                self.write_logfile(x*self.save_chunksize)

        self.get_motion(rest)

        if(self.do_write_logfile == True):
            self.additional_logmessage += "FINISHED SUCCESS"
            self.write_logfile(-1)

        self.write_hdf5()
        self.start_pos = 0#Don't change the objects initial state

        print("FINISHED")

    def get_motion(self, n_pairs_to_read, start_pos = None): #to read all frames n_pairs_to_read shall be total_frames of video
        if(self.n_pairs_to_read == None):
            self.n_pairs_to_read = n_pairs_to_read
        if(self.absolute_start_pos == None):#For reporting/log only
            if(start_pos == None):
                self.absolute_start_pos = 0
            else:
                self.absolute_start_pos = start_pos# Use parametern only if not already set (by to_file())


        end_pair = self.start_pos + n_pairs_to_read

        if(start_pos!=None):
            self.cap.set(1,start_pos)
            self.start_pos = start_pos


        total_frames = self.total_frames

        self.buffer = []
        batch_size = 8
        frame = self.start_pos #Total framenumber. Actually frame nframes-1 in the end because always a pair of frames is processed
        data_left = True
        i = 1
        cycles = 1 #before printing
        old_frame = 0

        while(data_left):#if there is data left...
            if(not self.supress_output):
                if((i % cycles)==0):
                    old_frame = frame
                    begin = time.time()#To calculate how long it takes to compute 8 frames

            if(frame+batch_size > end_pair):#Get smaller batch if almost done; Slope stops due to second condition later
                current_size = end_pair - frame
                data_left, batch = self.get_batches_of_pairs(current_size)
                data_left = False
            else:# get full batch
                data_left, batch = self.get_batches_of_pairs(batch_size)#If accidently reaching end of file data_left will be false

            output = mp.Queue()
            # Setup a list of processes that we want to run

            if(self.algorithm=="farneback"):
                processes = [mp.Process(target=self.farneback_wrapper, args=(frame+pair, sample, output))
                            for sample, pair in zip(batch, range(len(batch)))]
            elif(self.algorithm=="brocks"):
                processes = [mp.Process(target=self.brocks_wrapper, args=(frame+pair, sample, output))
                            for sample, pair in zip(batch, range(len(batch)))]
            else:
                raise NotImplementedError

            if(self.use_multiprocessing): #Run in parallel, Run processes
                for p in processes:
                    p.start()

                # Get process results from the output queue
                results = [output.get() for p in processes]
                results.sort()
                results = [x[1] for x in results]
                self.buffer.extend(results)

                # Exit the completed processes
                for p in processes:
                    p.join()
            else:
                results = []

                if(self.algorithm == "farneback"):
                    for sample in batch:
                        results.append(self.farneback_wrapper(0, sample, None))
                elif(self.algorithm == "brocks"):
                    for sample in batch:
                        results.append(self.brocks_wrapper(0, sample, None))

                self.buffer.extend(results)

            frame += len(batch)

            if((i % cycles)==0):
                time_per_frame = 0
                if(not self.supress_output):
                    end = time.time()#To calculate how long it takes to compute 8 frames
                    try:
                        time_per_frame = (end-begin)/(frame - old_frame)
                    except:
                        self.additional_logmessage += "(frame - old_frame) was probably 0"
                        pass
                    print("Estimated time to compute " + str(self.n_pairs_to_read)+ " pairs: " + str((self.n_pairs_to_read*time_per_frame)) + " s ")
                    self.percent_completed = 100*((frame-self.absolute_start_pos)/self.n_pairs_to_read)
                    print(self.n_pairs_to_read)
                    print("## Completed " + str(self.percent_completed) + "% ##")
                    #print("## Current frame" + str(i*cycles*batch_size) + " ##")#i+1 as report after computation before increment
                    print("## Current frame" + str(self.cap.get(cv2.CAP_PROP_POS_FRAMES))+ " ##")
                self.total_time_estimate = self.n_pairs_to_read * time_per_frame
            i += 1


        self.start_pos = frame# Keep track in case being called once again without parameter
        self.buffer = np.array(self.buffer)
        return self.buffer
