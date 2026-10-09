"""Smoother and ordered CatBoost variants on the successful feature family."""
import hashlib
import json
from pathlib import Path

import smoking_push as workflow


def main():
    baseline=workflow.SPECS['baseline'].copy()
    workflow.SPECS={'baseline':baseline,
        'smooth5':dict(features='baseline',depth=5,learning_rate=.025,iterations=2600,l2_leaf_reg=10,random_strength=.2),
        'ordered6':dict(features='baseline',depth=6,learning_rate=.025,iterations=1800,l2_leaf_reg=5,
                        boosting_type='Ordered',random_strength=.5)}
    workflow.OUT=Path('outputs/ordered_smoking_push');workflow.OUT.mkdir(parents=True,exist_ok=True)
    identity=dict(source_sha256=hashlib.sha256(Path('ordered_smoking_push.py').read_bytes()).hexdigest())
    path=workflow.OUT/'ordered_identity.json'
    if path.exists():assert json.loads(path.read_text())==identity,'Ordered recipe changed; use another output directory.'
    else:path.write_text(json.dumps(identity,indent=2))
    workflow.main()


if __name__=='__main__':main()
