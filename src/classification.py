"""Person-level held-out classification with training-fold imputation."""
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

def fit_loo(labels,features,names,ids,method,seed):
    labels=np.asarray(labels);features=np.asarray(features,float)
    n = len(labels)
    if len(set(ids)) != n: raise ValueError('Aggregate repeated visits before LOPO')
    if set(labels.tolist()) != {0, 1} or min(np.bincount(labels.astype(int))) < 2:
        raise ValueError('LOPO requires at least two people per class; the two-person demo cannot estimate AUC')
    if features.shape != (n, len(names)) or len(ids) != n: raise ValueError('Feature/identity mismatch')
    result=np.empty(n);folds=[];coef=[]
    for held in range(n):
        mask=np.arange(n)!=held;train=features[mask].copy();test=features[held:held+1].copy()
        active=[];fill=[]
        for j,name in enumerate(names):
            valid=np.isfinite(train[:,j]);values=train[valid,j]
            if not len(values):continue
            if name.endswith('_phase'):
                z=np.mean(np.exp(1j*np.deg2rad(values)))
                if abs(z)<=1e-12:continue
                value=float(np.rad2deg(np.angle(z)))
            else:value=float(np.median(values))
            active.append(j);fill.append(value)
        if not active:raise RuntimeError('No valid training-fold features; do not fabricate OOF probabilities')
        tr=train[:,active];te=test[:,active];imputed_train=int((~np.isfinite(tr)).sum());imputed_test=int((~np.isfinite(te)).sum())
        fills=np.array(fill);tr=np.where(np.isfinite(tr),tr,fills);te=np.where(np.isfinite(te),te,fills)
        scaler=StandardScaler().fit(tr);assert scaler.n_samples_seen_==n-1
        clf=LogisticRegression(C=1.,penalty='l2',solver='lbfgs',tol=1e-4,max_iter=5000,class_weight=None,random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning);clf.fit(scaler.transform(tr),labels[mask])
        result[held]=clf.predict_proba(scaler.transform(te))[0,1]
        folds.append(dict(method=method,seed=seed,person_id=ids[held],train_people=n-1,
                          active_features=';'.join(names[j] for j in active),dropped_features=';'.join(names[j] for j in range(len(names)) if j not in active),
                          imputed_training_values=imputed_train,imputed_heldout_values=imputed_test,intercept=float(clf.intercept_[0])))
        coef.extend(dict(method=method,seed=seed,held_out=ids[held],feature=names[j],coefficient=float(clf.coef_[0,k]),
                    imputation_value=float(fills[k]),train_mean=float(scaler.mean_[k]),train_scale=float(scaler.scale_[k])) for k,j in enumerate(active))
    if not np.isfinite(result).all():raise RuntimeError('Incomplete OOF predictions')
    return result,folds,coef
