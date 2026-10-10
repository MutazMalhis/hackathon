"""Complementary XGBoost classifiers under the same development checks."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import xgboost
from xgboost import XGBClassifier

import smoking_push as workflow


class CategoricalClassifier:
    def __init__(self,model,categories,columns):
        self.model=model;self.categories=categories;self.columns=columns

    def transform(self,x):
        result=x[self.columns].copy()
        for column,values in self.categories.items():
            result[column]=pd.Categorical(result[column],categories=values)
        return result

    def predict_proba(self,x):return self.model.predict_proba(self.transform(x))

    def save_model(self,path):
        output=Path(path).with_suffix('.ubj');self.model.save_model(output)
        output.with_suffix('.categories.json').write_text(json.dumps(dict(categories=self.categories,columns=self.columns)))


def main():
    baseline=workflow.SPECS['baseline'].copy()
    workflow.SPECS={'baseline':baseline,
        'xgb3':dict(features='baseline',max_depth=3,learning_rate=.04,n_estimators=1400,reg_lambda=10,min_child_weight=5,drop_ids=True),
        'xgb4':dict(features='baseline',max_depth=4,learning_rate=.03,n_estimators=1800,reg_lambda=15,min_child_weight=10,drop_ids=False)}
    original_fit=workflow.fit
    def fit(reference,test,name,seed):
        if name=='baseline':return original_fit(reference,test,name,seed)
        spec=workflow.SPECS[name]
        x=workflow.inputs(reference,reference,test,'baseline',training=True)
        if spec['drop_ids']:x=x.drop(columns=['Workplace_ID','Urine_Plate','POC_Batch'])
        categories={c:sorted(x[c].unique().tolist()) for c in workflow.CATS if c in x}
        for column,values in categories.items():x[column]=pd.Categorical(x[column],categories=values)
        model=XGBClassifier(tree_method='hist',enable_categorical=True,objective='binary:logistic',
            eval_metric='auc',n_jobs=4,random_state=seed,subsample=.9,colsample_bytree=.9,
            **{k:v for k,v in spec.items() if k not in ['features','drop_ids']})
        model.fit(x,reference.Smoking)
        return CategoricalClassifier(model,categories,x.columns.tolist())
    workflow.fit=fit
    workflow.OUT=Path('outputs/xgboost_smoking_push');workflow.OUT.mkdir(parents=True,exist_ok=True)
    identity=dict(source_sha256=hashlib.sha256(Path('xgboost_push.py').read_bytes()).hexdigest(),xgboost_version=xgboost.__version__)
    path=workflow.OUT/'xgboost_identity.json'
    if path.exists():assert json.loads(path.read_text())==identity,'XGBoost recipe changed; choose another output directory.'
    else:path.write_text(json.dumps(identity,indent=2))
    workflow.main()


if __name__=='__main__':main()
