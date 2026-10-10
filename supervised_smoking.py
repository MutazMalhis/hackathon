"""Label-adjusted urine plate offsets, with exact leave-one-out training features.

Uses labels from fitting rows only. Each row's own label is excluded from both
the fitted marker/Smoking coefficient and its plate offset.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import smoking_push as workflow


def marker_feature(reference,frame,training=False):
    r=np.log(reference.Urine_Cotinine_ng_mL/reference.Urine_Creatinine)
    valid=r.notna()&reference.Urine_Plate.notna()
    work=pd.DataFrame({'plate':reference.Urine_Plate[valid],'r':r[valid],
                       's':reference.Smoking[valid].astype(float)})
    work['rs']=work.r*work.s
    grouped=work.groupby('plate').agg(n=('r','size'),r=('r','sum'),s=('s','sum'),rs=('rs','sum'))
    cg=grouped.rs-grouped.r*grouped.s/grouped.n
    vg=grouped.s-grouped.s**2/grouped.n
    cross=float(cg.sum());variance=float(vg.sum())
    beta=cross/max(variance,1e-9)
    n=frame.Urine_Plate.map(grouped.n).to_numpy(dtype=float)
    sr=frame.Urine_Plate.map(grouped.r).to_numpy(dtype=float)
    sy=frame.Urine_Plate.map(grouped.s).to_numpy(dtype=float)
    marker=np.log(frame.Urine_Cotinine_ng_mL/frame.Urine_Creatinine).to_numpy()
    if training:
        assert frame.ID.tolist()==reference.ID.tolist()
        y=reference.Smoking.to_numpy(dtype=float)
        included=valid.to_numpy()
        oldc=frame.Urine_Plate.map(cg).fillna(0).to_numpy();oldv=frame.Urine_Plate.map(vg).fillna(0).to_numpy()
        remainder_n=n-1;remainder_r=sr-marker;remainder_y=sy-y
        xy=frame.Urine_Plate.map(grouped.rs).to_numpy()-marker*y
        with np.errstate(divide='ignore',invalid='ignore'):
            newc=np.where(remainder_n>0,xy-remainder_r*remainder_y/remainder_n,0.)
            newv=np.where(remainder_n>0,remainder_y-remainder_y**2/remainder_n,0.)
        numerator=cross-np.where(included,oldc-newc,0.)
        denominator=variance-np.where(included,oldv-newv,0.)
        slope=numerator/np.maximum(denominator,1e-9)
        n=n-included;sr=sr-np.where(included,marker,0.);sy=sy-np.where(included,y,0.)
    else:slope=np.full(len(frame),beta)
    # Keep the same fallback for unknown plates; remove own row from the global
    # fallback too when constructing a training feature.
    global_valid=r.notna().to_numpy();all_n=int(global_valid.sum());all_r=float(r.sum());all_y=float(reference.Smoking[r.notna()].sum())
    if training:
        y=reference.Smoking.to_numpy(dtype=float)
        global_mean=(all_r-np.where(global_valid,marker,0.)-slope*(all_y-np.where(global_valid,y,0.)))/np.maximum(all_n-global_valid,1)
    else:global_mean=(all_r-slope*all_y)/max(all_n,1)
    with np.errstate(divide='ignore',invalid='ignore'):
        offset=(sr-slope*sy)/n
    offset=np.where(np.isfinite(offset),offset,global_mean)
    return pd.DataFrame({'label_adjusted_cotinine':marker-offset,
                         'label_adjusted_plate_n':np.nan_to_num(n,nan=0.)},index=frame.index)


def main():
    original_inputs=workflow.inputs
    def extended_inputs(reference,frame,test,variant,training=False):
        x=original_inputs(reference,frame,test,'baseline',training)
        if variant=='supervised':x=x.join(marker_feature(reference,frame,training))
        return x
    workflow.inputs=extended_inputs
    baseline=workflow.SPECS['baseline'].copy()
    workflow.SPECS={'baseline':baseline,'supervised':{**baseline,'features':'supervised'},
                    'supervised4':dict(features='supervised',depth=4,learning_rate=.035,iterations=2200,l2_leaf_reg=5)}
    workflow.OUT=Path('outputs/supervised_smoking_push');workflow.OUT.mkdir(parents=True,exist_ok=True)
    path=workflow.OUT/'supervised_identity.json'
    identity={'source_sha256':hashlib.sha256(Path('supervised_smoking.py').read_bytes()).hexdigest()}
    if path.exists():assert json.loads(path.read_text())==identity,'Supervised recipe changed; choose a new output directory.'
    else:path.write_text(json.dumps(identity,indent=2))
    workflow.main()


if __name__=='__main__':main()
