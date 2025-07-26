from hmmlearn import hmm
import numpy as np

__all__ = ["hmm_sampling"]

def hmm_sampling(dim:int=3, n_data:int=500, trs:float=0.05, sigma:float=0.5, rand:int=None, 
                 ans:bool=False, scale:float=1, poi:bool=False):
    """
    Sampline function.py

    Parameters
    ----------
    dim : int, default is 3
        The number of states.
    n_data : int, default is 500
        The length of data.
    trs : float, default is 0.05
        Probability of transition.
    sigma : float, default is 0.5
        Standard deviation of noise.
    rand : int or None, optional
        Random seed.
    ans : bool, default is False
        If the answer of state sequence is returned.
    scale : int, default is 1
        Interval between mean values.
    poi : bool, default is False
        If Poisson distributed.

    """    
    startprob= np.full(dim, 1.0/dim)
    transmat = np.full((dim,dim), trs/(dim-1)) + np.identity(dim)*(1.0 - trs/(dim-1) - trs)
    means = np.arange(1, dim+1).reshape(-1,1)*scale
    covars = np.full(dim, sigma*sigma)*scale*scale
    
    model = hmm.GaussianHMM(n_components=dim, covariance_type="spherical")
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.means_ = means
    model.covars_ = covars
    
    data, states = model.sample(n_data,random_state=rand)
    
    answer = model.means_[states, 0]

    if poi:
        np.random.seed(rand)
        data_1 = np.random.poisson(lam=answer)
        np.random.seed()
    else:
        data_1 = data.flatten()

    if ans:
        return data_1, answer
    else:
        return data_1
