"""Compare LightGBM, Extra Trees, and an RBF SVM with the submitted classifier.

Reuse the existing development split and promotion gates. No Kaggle upload.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import shutil

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

import smoking_push as workflow


class AlternativeClassifier:
    def __init__(self, estimator, columns, categories=None, logged=None):
        self.estimator = estimator
        self.columns = columns
        self.categories = categories or {}
        self.logged = logged or []

    def transform(self, frame):
        x = frame.loc[:, self.columns].copy()
        for column, values in self.categories.items():
            x[column] = pd.Categorical(x[column], categories=values)
        for column in self.logged:
            x[column] = np.log1p(x[column].clip(lower=0))
        return x

    def predict_proba(self, frame):
        return self.estimator.predict_proba(self.transform(frame))

    def save_model(self, path):
        # Store estimator and feature contract without pickling the wrapper class.
        joblib.dump(dict(estimator=self.estimator, columns=self.columns,
                         categories=self.categories, logged=self.logged), path+'.joblib')

    @classmethod
    def load_model(cls, path):
        return cls(**joblib.load(path+'.joblib'))


def fit_alternative(reference, test, name, seed):
    spec = workflow.SPECS[name]
    x = workflow.inputs(reference, reference, test, 'baseline', training=True)
    if name.startswith('lgb'):
        categories = {c: sorted(x[c].unique().tolist()) for c in workflow.CATS}
        model = LGBMClassifier(objective='binary', n_jobs=4, random_state=seed,
            verbosity=-1, deterministic=True, force_col_wise=True,
            subsample=.85, subsample_freq=1, colsample_bytree=.85,
            **{k:v for k,v in spec.items() if k!='features'})
        wrapper = AlternativeClassifier(model, x.columns.tolist(), categories)
    else:
        x = x.drop(columns=['Workplace_ID', 'Urine_Plate', 'POC_Batch'])
        categorical = [c for c in workflow.CATS if c in x]
        numeric = [c for c in x if c not in categorical]
        # Fixed transforms; columns selected using fitting rows only.
        logged = [c for c in numeric if x[c].min() >= 0]
        preprocessing = ColumnTransformer([
            ('numeric', make_pipeline(SimpleImputer(strategy='median', add_indicator=True),
                                       StandardScaler()), numeric),
            ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)])
        if name=='extra':
            learner = ExtraTreesClassifier(n_estimators=700, min_samples_leaf=3,
                max_features=.8, random_state=seed, n_jobs=4)
        else:
            learner = SVC(C=3., gamma=.015, kernel='rbf', probability=True,
                          random_state=seed, cache_size=512)
        wrapper = AlternativeClassifier(make_pipeline(preprocessing, learner),
                                        x.columns.tolist(), logged=logged)
    wrapper.estimator.fit(wrapper.transform(x), reference.Smoking)
    return wrapper


def select():
    _, test, _, dev = workflow.data()
    predictions, _ = workflow.cv_predictions(dev, test, list(workflow.SPECS),
                                             41009, workflow.OUT/'selection')
    rows = []
    for name in workflow.SPECS:
        for weight in ([1.] if name=='baseline' else [.15, .3, .5, 1.]):
            p = weight*predictions[name]+(1-weight)*predictions['baseline']
            rows.append(dict(model=name, weight=weight,
                             auc=float(workflow.roc_auc_score(dev.Smoking, p))))
    pd.DataFrame(rows).to_csv(workflow.OUT/'selection.csv', index=False)
    config = dict(selected=max(rows,key=lambda r:r['auc']),baseline_auc=rows[0]['auc'],
                  specifications=workflow.SPECS, selected_before_confirmation=True,
                  holdout_used_for_selection=False)
    (workflow.OUT/'frozen_config.json').write_text(json.dumps(config,indent=2))
    print('SELECTED', json.dumps(config), flush=True)


def main():
    workflow.OUT = Path('outputs/alternative_models');workflow.OUT.mkdir(parents=True,exist_ok=True)
    baseline = workflow.SPECS['baseline'].copy()
    workflow.SPECS = {'baseline': baseline,
        'lgb15':dict(features='baseline',num_leaves=15,learning_rate=.025,n_estimators=1600,
                     min_child_samples=30,reg_lambda=5.,reg_alpha=.1),
        'lgb31':dict(features='baseline',num_leaves=31,learning_rate=.025,n_estimators=1200,
                     min_child_samples=40,reg_lambda=10.,reg_alpha=.5),
        'extra':dict(features='baseline'), 'svm':dict(features='baseline')}
    identity = dict(code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    versions={p:importlib.metadata.version(p) for p in ['lightgbm','scikit-learn','joblib']})
    path = workflow.OUT/'alternative_identity.json'
    if path.exists():
        assert json.loads(path.read_text())==identity, 'Recipe changed; use a new output directory.'
    else:
        path.write_text(json.dumps(identity,indent=2))
    # Reuse baseline predictions only after verifying the exact parent recipe/data.
    old_dir = Path('outputs/smoking_push_consistent')
    old_identity = json.loads((old_dir/'run_identity.json').read_text())
    for p,h in {**old_identity['code'],**old_identity['data']}.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    for stage in ['selection','confirmation']:
        dest=workflow.OUT/stage;dest.mkdir(exist_ok=True)
        for source in (old_dir/stage).glob('baseline_*.csv'):
            if not (dest/source.name).exists():shutil.copyfile(source,dest/source.name)
    original_fit=workflow.fit
    workflow.fit=lambda ref,test,name,seed: (original_fit(ref,test,name,seed)
        if name=='baseline' else fit_alternative(ref,test,name,seed))
    workflow.select=select
    workflow.main()


if __name__=='__main__':
    main()
