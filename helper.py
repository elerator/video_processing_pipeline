import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import json

import re

from scipy import stats

def combined_trials_for_dyad(folder,eeg_filenames,vmrks,participant, exclude, electrodes):
    """ Retrieves the trials periods from the three eeg files and appends them.
    Params:
        folder: folder path (string)
        eeg_filenames: filenames(string)
        vmrks: dictionary containing information from a vmrk,
        participant: 0 or 1
        exclude: list of electrode indices to exclude
        electrodes: List of names of electrodes including those that are to be excluded

    Returns:
        Dictionary mapping channels to data of all runs
    """
    powers = []
    for filename, vmrk in zip(eeg_filenames,vmrks):
        power = frequency_bands_to_common_reference(folder+filename,electrodes
                                                            ,participant =0, exclude = exclude)
        times = get_trial_times(vmrk)
        trials_only(power, times)
        powers.append(power)
    return extend_nparray_keys(extend_nparray_keys(powers[0],powers[1]),powers[2])

def extend_nparray_keys(d1,d2):
    """ Extends keys in a dictionary that are 1D numpy arrays and returns the new Dictionary
        Params:
            d1: Dictionary mapping keys to 1d numpy arrays
            d2: Dictionary mapping keys to 1d numpy arrays
        Returns:
            Dictionary mapping keys to extended 1d numpy arrays
    """
    dr = {}
    for k,v in d1.items():
        l = list(d1[k])
        l.extend(list(d2[k]))
        dr[k] = np.array(l)
    return dr

def trials_only(power,times):
    """ Cuts the trial periods inplace (i.e. both runs) from a single eeg/motion file appends them and returns combined ndarray
        Args:
            power: Dictionary mapping strings(Electodes) to data (1d np array)
            times: times of the runs
    """

    for k,v in power.items():
        t1 = times[0,0]//20
        t2 = times[0,1]//20
        t3 = times[1,0]//20
        t4 = times[1,1]//20

        l1 = list(v[t1:t2])
        l2 = list(v[t3:t4])
        l1.extend(l2)

        assert len(l1) == np.abs((t2-t1))+np.abs((t4-t3))
        power[k] = np.array(l1)

def get_trial_times(parsed_vmrk):
    """ Uses the information from a parsed vmrk to returns the times for the trial period (beginning and end).
        Times are computed according to the eyes open trigger (begin) and eyes closed (end) triggerself.
        Args:
            parsed_vmrk: Dictionary containing the information from a markerfile
    """
    start = get_times_of(parsed_vmrk,"S 20")#end refers to end of eyes closed. start to start. We look for reverse
    start.append(None)
    end = [None]
    end.extend(get_times_of(parsed_vmrk,"S 21"))
    times = np.array([end,start]).T
    times = times[1:-1]
    return times

def get_times_of(parsed, event, pois = None):
    """ Retrieves times for events given a parsed vmrkself
    Args:
        parsed: A dictionary containing the information from a vmrks
        event: String specifying the event name e.g. S128
        pois: Period of interest. If given results are added to it (list)

    """
    if not pois:
        pois = []
    last_idx = 0

    while(True):
        try:
            remain = list(np.array(parsed['description'])[last_idx:])

            last_idx = last_idx + remain.index(event)
            #print(last_idx)

            pois.append(int(parsed["time"][last_idx]))
            last_idx += 1

        except Exception as e:
            #print(e)#if there is no more entry to be found via .index break, return periods of interest
            break
    return pois

def load_eeg(filepath, channel=None, participant = -1):
    """ Loads eeg file, emits channeldata and channel_signal
    Args:
        filepath: The path to a eeg file
        channel: The channel that is to be loaded and returned
    Returns:
        A 1d Numpy array containing the eeg data for one channel
    """
    if type(channel)==type(None):
        eegs = []
        if participant <= -1:#no participant provided
            for electrode in range(64):
                eegs.append(load_eeg(filepath,channel = electrode))
        else:
            for electrode in range(32):
                eegs.append(load_eeg(filepath,electrode,participant*32))
        return np.array(eegs)

    n_channels = 64
    bytes_per_sample = 2 #Because int16

    my_type = np.dtype([("channel"+str(x),np.int16) for x in range(0,n_channels)])
    byte_size = os.path.getsize(filepath)

    nFrames =  byte_size // (bytes_per_sample * n_channels);
    data = np.array(np.fromfile(filepath,dtype=my_type))["channel"+str(channel)]

    data = np.array(data, dtype= np.float32)
    data[data==32767] = np.nan
    data[data==-32768] = np.nan
    data[data==-32767] = np.nan

    return data



def spectrogram(eeg, f_max =100, f_min = 0, window_size =2000,step_size = 20):
    """ Computes a time frequency spectrogram for an eeg using fast fourier transform
    Params:
        f_max: maximal frequency of the spectrogram
        f_min: minimal frequency
        window_size: Size of sliding window that is used to split data and transfer it to fft
        step_size: The size of the steps the sliding window makes in skipping through the eeg
    """
    output_size = (len(eeg)//step_size,f_max)
    power = np.zeros(output_size)

    for i in range(len(power)):
        snippet = eeg[step_size*i-window_size//2:step_size*i+window_size//2]
        if not len(snippet == window_size):
            continue
        val = np.fft.fft(snippet).real[:f_max]
        power[i] = val

        if i % 2000 == 0:
            print(".", end ="")
    return power.T[f_min:]


def parse_vmrk(path):
    """ Parses vmrk file and returns a dictionary containing the information.
        The keys denote the kind of data whereas the values are a dictionary
    """

    with open(path) as f:
        content = f.readlines()

    data = {'marker number':[], 'type':[], 'description':[], 'time':[], 'size':[], 'channel':[]}

    entry = 0
    for line in content:
        match = re.match("Mk", line)
        if(match != None):
            markers = re.search("[0-9][0-9]?", line)
            data["marker number"].append(int(markers.group(0)))
            line = line[markers.end():]#use rest of line only next

            markers = re.match("(.*?),",line)
            data["type"].append(markers.group(1)[1:])#Group 1 is exclusive , while group 0 is inclusive ,
            line = line[markers.end():]

            markers = re.search("(.*?),",line)
            data["description"].append(markers.group(1))
            line = line[markers.end():]

            markers = re.search("(.*?),",line)
            data["time"].append('0' + markers.group(1))# '0' + is necessary as some fields are empty
            line = line[markers.end():]

            markers = re.search("(.*?),",line)
            data["size"].append(int('0' + markers.group(1)))
            line = line[markers.end():]

            try:#In the first line there is an additional value we dont want to parse
                data["channel"].append(int('0' + line))
            except:
                data["channel"].append(0)
    return data


def weighted_histograms(data, roi = None,frame_range= None, n_bins = 36):
    """ Computes weighted histograms from a motion 4d motion tensor
    data: A motion tensor of shape framewidth x frameheigt x frames x 2 containing x and y components
            for motion vectors for each pixel and frame
    roi: A list containing dictionaries mapping the values "y1", "x1", "y2" and "x2" to the respective coordinates
        of a region of interest that is analyzed to retrieve weigted weighted_histograms
    frame range: Defines the range of frames to be processed
    n_bins: The number of directions (bins) the motion information is accumulated for. Standard: 36
    """
    if(360%n_bins != 0):
        raise Exception("Provide a number of bins such that 360%bins == 0 (e.g. 36, or 72)")
    bin_width = 360//n_bins

    if frame_range:
        data = data[frame_range[0]:frame_range[1]]

    outary = np.ndarray((data.shape[0],n_bins))

    current_roi = {"y1":0,"x1":0,"y2":data.shape[1],"x2":data.shape[2]}#xy xy full frame by default
    for d, i in zip(data, range(data.shape[0])):
        if(roi):
            try:#If there is info for frame update the region of interest
                roi[str(i)]
                current_roi["y1"] = roi[str(i)]["y1"]//16
                current_roi["x1"] = roi[str(i)]["x1"]//16
                current_roi["y2"] = roi[str(i)]["y2"]//16
                current_roi["x2"] = roi[str(i)]["x2"]//16
            except:
                pass
        d = d[current_roi["y1"]:current_roi["y2"],current_roi["x1"]:current_roi["x2"],:]
        ary = np.array([d[:,:,0].flatten(),d[:,:,1].flatten()])
        deg = np.degrees(np.arctan2(ary[0],ary[1]))#.reshape(d[0].shape[0:2])
        length = np.linalg.norm(ary, axis = 0)#.reshape(d[0].shape[0:2])
        bins = (np.arange(n_bins+1)*bin_width)-180 #start at -170. Go up in steps of 10 to 180
        weightedHist, bin_edges = np.histogram(deg, weights = length, bins = bins)
        outary[i] = weightedHist
        if i%100 == 0:
            print(i/data.shape[0])
    return outary.T


def interpolate_missing_frames(mothistmap):
    """ May be used to interpolate missing frames of motion histogram maps
    Params:
        mothistmap: A 2d numpy array containing motion information for different directions and times
    """
    for i in range(mothistmap.shape[1]):#median
        if(np.all(mothistmap[:,i]==0)):
            print(i)
            try:
                mothistmap[:,i] = (mothistmap2[:,i-1]+mothistmap2[:,i+1])/2
            except:
                pass


def frequency_bands_to_common_reference(path, electrode_names= None,
                                            participant = 0, vmrk = None,channel=None, exclude=[]):
    reference = None
    time = 0

    freq_alpha = [8,14]#Cut frequencies from fft spectrogram
    freq_beta = [14,30]
    freq_gamma_low = [30,50]
    freq_gamma_high = [50,100]

    alpha_contrasts = []
    beta_contrasts = []
    gamma_low_contrasts = []
    gamma_high_contrasts = []

    dictionary = {}

    if vmrk:#Vmrk provided? if so cut off initial part i.e. set time do do that later
        if type(vmrk) == type(""):#If vmrk is a path parse it before looking for time
            idx = [x == "S128" for x in parse_vmrk(vmrk)['description']].index(True)
            time = int(parse_vmrk(vmrk)['time'][idx])//20

        else:# no need to parse
            idx = [x == "S128" for x in vmrk['description']].index(True)
            time = int(vmrk['time'][idx])//20

    if not channel:
        reference = np.nanmean(load_eeg(path,participant=participant),axis = 0)
        print(".", end = "")
    else:
        reference = load_eeg(path, channel)[participant*32:(participant*32)+32]

    for electrode in range(32*participant,32*participant+32):
        if electrode in exclude:
            continue
        print(".", end="")
        current_signal = load_eeg(path, electrode)-reference
        spectrum = np.abs(spectrogram(current_signal, f_min=0,f_max=100))
        if not electrode_names:
            alpha_contrasts.append(np.nansum(spectrum[freq_alpha[0]:freq_alpha[1],time:],axis=0))
            beta_contrasts.append(np.nansum(spectrum[freq_beta[0]:freq_beta[1],time:],axis=0))
            gamma_low_contrasts.append(np.nansum(spectrum[freq_gamma_low[0]:freq_gamma_low[1],time:],axis=0))
            gamma_high_contrasts.append(np.nansum(spectrum[freq_gamma_high[0]:freq_gamma_high[1],time:],axis=0))
        else:
            dictionary["alpha_"+electrode_names[electrode]] = np.nansum(spectrum[freq_alpha[0]:freq_alpha[1],time:],axis=0)
            dictionary["beta_"+electrode_names[electrode]] = np.nansum(spectrum[freq_beta[0]:freq_beta[1],time:],axis=0)
            dictionary["gamma1_"+electrode_names[electrode]] = np.nansum(spectrum[freq_gamma_low[0]:freq_gamma_low[1],time:],axis=0)
            dictionary["gamma2_"+electrode_names[electrode]] = np.nansum(spectrum[freq_gamma_high[0]:freq_gamma_high[1],time:],axis=0)

    if not electrode_names:
        return np.array([alpha_contrasts, beta_contrasts, gamma_low_contrasts, gamma_high_contrasts])
    else:
        return dictionary


def frequency_bands_between_electrodes(path, participant = 0, vmrk = None):
    """ Computes the contrast between each pair of electrodes and returns the bandpowers
        for the alpha, beta as well as for low and high gamma frame_range
        If a vmrk is provided the start trigger is for the video is found and the data stripped accordingly
        Args:
            path: Path to eeg file
            participant: 0 or 1. If 0 data for participant 0 is processed, if 1 for the second participant.
    """
    time = 0
    if vmrk:
        idx = [x == "S128" for x in parse_vmrk(vmrk)['description']].index(True)
        time = int(parse_vmrk(vmrk)['time'][idx])//20

    alpha_contrasts = np.ndarray((32,32),dtype=np.object)
    beta_contrasts = np.ndarray((32,32),dtype=np.object)
    gamma_low_contrasts = np.ndarray((32,32),dtype=np.object)
    gamma_high_contrasts = np.ndarray((32,32),dtype=np.object)

    freq_alpha = [8,14]#Cut frequencies from fft spectrogram
    freq_beta = [14,30]
    freq_gamma_low = [30,50]
    freq_gamma_high = [50,100]


    i = 1
    for electrode_1 in range(32):
        for electrode_2 in range(32):
            if(electrode_1 < electrode_2):# Compute only half the matrix, avoid redundancy
                continue
            contrast = load_eeg(path,(participant*32)+electrode_1)-load_eeg(path,(participant*32)+electrode_2)
            spectrum = np.abs(spectrogram(contrast, f_min=0,f_max=100))
            print((1/((32*32)/2))*i)
            alpha_contrasts[electrode_1][electrode_2] = np.sum(spectrum[freq_alpha[0]:freq_alpha[1],time:],axis=0)
            beta_contrasts[electrode_1][electrode_2] = np.sum(spectrum[freq_beta[0]:freq_beta[1],time:],axis=0)
            gamma_low_contrasts[electrode_1][electrode_2] = np.sum(spectrum[freq_gamma_low[0]:freq_gamma_low[1],time:],axis=0)
            gamma_high_contrasts[electrode_1][electrode_2] = np.sum(spectrum[freq_gamma_high[0]:freq_gamma_high[1],time:],axis=0)
            i+=1
    return np.array([alpha_contrasts, beta_contrasts, gamma_low_contrasts, gamma_high_contrasts])

def correlations(bands, movement_signal, smoothing = False):
    """ Computes correlation coefficents for every electrode pair and a given movement channel_signal
    Args:
        bands:              N x N numpy array containing the bandpower-contrasts between each pair of electrodes (1D numpy arrays)
        movement_signal:    A 1D numpy array containing a movement channel_signal
        smoothing:          Boolean. If true smoothing is applied using a gaussian with sigma 50 for movement_signal
                            and sigma = 2 for eeg bandpowers
    """
    movement_signal = movement_signal[:bands[0,0].shape[0]]#cut movement_signal if its too long
    if(smoothing):#Smoothing
        movement_signal = ndimage.filters.gaussian_filter1d(movement_signal,50)

    correlations = []
    correlation_coefficients = np.zeros(shape=(32,32), dtype = np.float32)
    ps = np.zeros(shape=(32,32), dtype = np.float32)

    for i in range(32):
        for j in range(32):
            try:
                contrast = bands[i][j]

                if(smoothing):
                    contrast = ndimage.filters.gaussian_filter1d(contrast,2)


                slope, intercept, r, p, err = stats.linregress(contrast,movement_signal)
                correlation_coefficients[i][j] = r
                ps[i][j] = p
            except Exception as e:
                pass

            if(i % 100 == 0):
                print(".", end = "")

        correlations.append(correlation_coefficients)
        correlations.append(ps)
    correlations[0] = correlations[0]+correlations[0].T
    #correlations[1] = correlations[1]+correlations[1].T

    return np.array(correlations)


def correlation_shifted(row,row1,maxshift):
    """ Moves row relative to row1 and computes correlation
    Params:
        row:    1d datatrain
        row1:    1d datatrain
        maxshift: relative maximal shift in datapoints
    Returns:
        correlation for shift between minus and plut maxshift
    """
    minlen = np.min([len(row),len(row1)])#in case the rows are of different size
    corrs = []
    ps = []
    borders = np.arange(-maxshift,maxshift)
    for i in borders:
        #Start at maxshift for row such that there is data from shifted row1
        snippet1 = row[maxshift+i:minlen-maxshift+i]
        snippet2 = row1[maxshift:minlen-maxshift]
        slope, intercept, pearsons_r, p_value, std_err = stats.linregress(snippet1,snippet2)
        if(np.isnan(pearsons_r)):
            print("!", end = "")
            mask = ~np.isnan(snippet1) & ~np.isnan(snippet2)#avoid
            slope, intercept, pearsons_r, p_value, std_err = stats.linregress(snippet1[mask],snippet2[mask])


        corrs.append(pearsons_r)
        ps.append(p_value)
    return corrs, ps
