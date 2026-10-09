import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from alternative_models import AlternativeClassifier
from alternative_gamma import features


class AlternativeModelContractTests(unittest.TestCase):
    def test_gamma_features_ignore_held_out_targets(self):
        train=pd.read_csv(Path(__file__).resolve().parents[1]/'data/train.csv')
        reference=train.iloc[:50].copy();held=train.iloc[50:55].copy()
        expected=features(reference,held)
        held['Smoking']=1-held.Smoking
        held['Gamma_GT']=held.Gamma_GT*100
        pd.testing.assert_frame_equal(features(reference,held),expected)
        self.assertFalse({'Smoking','Gamma_GT','ID','POC_GGT','POC_Batch'} & set(expected.columns))

    def test_saved_model_preserves_predictions_for_unseen_category_and_missing_value(self):
        x = pd.DataFrame({'marker': np.arange(40, dtype=float),
                          'center': ['A', 'B']*20})
        wrapper = AlternativeClassifier(
            LGBMClassifier(n_estimators=25, num_leaves=4, min_child_samples=2,
                           verbosity=-1, n_jobs=1, random_state=42),
            ['marker', 'center'], {'center': ['A', 'B']})
        wrapper.estimator.fit(wrapper.transform(x), (x.marker>19).astype(int))
        held = pd.DataFrame({'center': ['NEW', 'A'], 'marker': [np.nan, 31.]})
        expected = wrapper.predict_proba(held)
        self.assertTrue(np.isfinite(expected).all())
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/'model')
            wrapper.save_model(path)
            loaded = AlternativeClassifier.load_model(path)
            np.testing.assert_allclose(loaded.predict_proba(held), expected, atol=0, rtol=0)


if __name__=='__main__':
    unittest.main()
