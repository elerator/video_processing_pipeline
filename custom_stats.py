import numpy as np

def steiger_ab_ac_corr_significance(train1,train2,train_common):
    """
    Checks if correlations between train1 and train_common are different from correlations between train2 and train_common.
    Params:
        train1: 1D datatrain
        train2: 1D datatrain
        train_common: The 1D datatrain train1 and train2 correlate with and for which it is shown if they are significantly different
    Returns:
        Dictionary containing keys 'z_score', 'one_tail_p' and 'two_tail_p' mapping to the respective values
    """

    if not (len(train1)==len(train2) or len(train2) ==len(train_common) or len(train1) == len(train_common)):
        raise Exception("Data of different length provided")
    n = len(train1)
    r_jk = np.corrcoef(train1,train_common)[1,0]
    r_jh = np.corrcoef(train2,train_common)[1,0]
    r_kh = np.corrcoef(train1,train2)[1,0]

    result = compare_jk_and_jh_regression(r_jk,r_jh,,r_kh,n)
    result["r_ab"] = r_jk
    result["r_cb"] = r_jh
    result["r_ac"] = r_kh
    return(result)

def compare_jk_and_jh_regression(r_jk,r_jh,r_kh, n):
  """ Computes z-score and one/two-tailed p values for two correlations (r_jk and r_jh) with one common variable for the same dataset.
      Steiger, J. H. (1980). Tests for comparing elements of a correlation matrix. Psychological Bulletin, 87, 245-251.
      Translated from javascript: http://quantpsy.org/corrtest/corrtest2.htm
  Params:
    r_jk: First correlation to be compared
    r_jh: Second correlation to be compared
    r_kh: Correlation between variables that are not the common one
  Rerurns:
    Dictionary containing keys 'z_score', 'one_tail_p' and 'two_tail_p' mapping to the respective values

  """
  nn1 = n
  r1val = r_jk
  r2val = r_jh
  r3val = r_kh
  x1=np.sum([1,r1val])
  x2=np.sum([1,((-1)*r1val)])
  y1=np.sum([1,r2val])
  y2=np.sum([1,((-1)*r2val)])
  zz1=0.5*(np.log(x1)-np.log(x2))
  zz2=0.5*(np.log(y1)-np.log(y2))

  cov12=(r3val*(1-(np.power(r1val,2))-(np.power(r2val,2)))-0.5*r1val*r2val*(1-(np.power(r1val,2))-(np.power(r2val,2))-(np.power(r3val,2))))/((1-(np.power(r1val,2)))*(1-(np.power(r2val,2))))

  zz=(zz1-zz2)/(np.sqrt((2-2*cov12)/(nn1-3)))
  pp2=Norm(zz)
  pp1=Norm(zz)/2

  print(zz)
  print(pp1)
  print(pp2)
  return {"z_score":zz,"one_tail_p":pp1,"two_tail_p":pp2}


def ChiSq(x,n):
    if n == 1 and x>1000:
        return 0
    if x > 1000 or n > 1000:
        q = ChiSq((x-n)*(x-n)/(2*n),1)/2
        if(x>n):
            return q
        else:
            return 1-q
    p = np.exp(-0.5*x)

    if((n%2)==1):
        p=p*np.sqrt(2*x/np.pi)
    k = n
    while(k>=2):
        p=p*x/k
        k=k-2
    t = p
    a = n
    while(t>0.0000000001*p):
        a=a+2; t=t*x/a
        p=p+t
    return 1-p


def Norm(z):
    return ChiSq(z*z,1)
