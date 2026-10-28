import numpy as np
from sklearn.feature_selection import f_classif

class TrainPreprocessor:
    """Every learned statistic is derived solely from supplied real training rows."""
    def __init__(self, k=50, missingness=.25, variance=1e-8):
        self.k,self.missingness,self.variance=k,missingness,variance

    def fit(self,x,y,names):
        x=np.asarray(x,dtype=np.float64)
        self.keep=np.flatnonzero(np.isnan(x).mean(axis=0)<=self.missingness)
        z=x[:,self.keep]
        self.median=np.nanmedian(z,axis=0)
        z=np.where(np.isnan(z),self.median,z)
        valid=np.flatnonzero(z.var(axis=0)>self.variance)
        scores=f_classif(z[:,valid],y)[0]
        scores=np.nan_to_num(scores,nan=-np.inf,posinf=np.finfo(float).max)
        order=np.lexsort((np.asarray(names)[self.keep[valid]],-scores))
        if len(order)<self.k: raise ValueError('Too few surviving features')
        self.selected=valid[order[:self.k]]
        self.names=np.asarray(names)[self.keep[self.selected]]
        self.n_fit=len(x)
        return self

    def transform(self,x):
        z=np.asarray(x,dtype=np.float64)[:,self.keep]
        return np.where(np.isnan(z),self.median,z)[:,self.selected]
