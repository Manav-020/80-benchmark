import numpy as np
from sklearn.metrics import confusion_matrix,roc_auc_score

def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p)
    tn,fp,fn,tp=confusion_matrix(y,p>=.5,labels=[0,1]).ravel()
    sensitivity=tp/(tp+fn) if tp+fn else None
    specificity=tn/(tn+fp) if tn+fp else None
    # BA undefined on a single-class held-out chip; report class recall instead.
    ba=(sensitivity+specificity)/2 if sensitivity is not None and specificity is not None else None
    return dict(n=len(y),balanced_accuracy=ba,accuracy=float(((p>=.5)==y).mean()),
        sensitivity=sensitivity,specificity=specificity,roc_auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
        tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),benchmark_80=bool(ba>=.8) if ba is not None else None)
