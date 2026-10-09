"""Numerical calibration and nested cross-fitting isolation checks."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from advanced_calibration import fit_calibration, calibrated_prediction
from experiments import batch_correct
from cross_target import smoking_features


class CalibrationTests(unittest.TestCase):
    def reference(self):
        x = np.tile([4., 8., 16., 32.], 3)
        batch = np.repeat([1, 2, 3], 4)
        return pd.DataFrame({'POC_GGT':x,'POC_Batch':batch,
                             'Gamma_GT':np.exp(.2*batch)*x**1.2})

    def test_within_batch_slope_recovery(self):
        ref = self.reference()
        config = dict(fit_slope=True, exclude_floor=True, low_weight=1, weight=1)
        state = fit_calibration(ref, config)
        self.assertAlmostEqual(state.slope,1.2,places=12)
        frame = pd.DataFrame({'POC_GGT':[6.,12.,24.], 'POC_Batch':[1,2,3]})
        p,known,floor = calibrated_prediction(state,frame,np.full(3,20.),config)
        np.testing.assert_allclose(p,np.exp(.2*np.arange(1,4))*frame.POC_GGT**1.2)
        self.assertTrue(known.all());self.assertFalse(floor.any())

    def test_original_calibration_is_reproduced(self):
        ref = self.reference();frame=ref.copy();fallback=np.full(len(frame),20.)
        config=dict(fit_slope=False,exclude_floor=False,low_weight=1,weight=.9)
        state=fit_calibration(ref,config)
        actual,_,_=calibrated_prediction(state,frame,fallback,config)
        expected,_=batch_correct(ref,frame,fallback,weight=.9)
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)

    def test_floor_rows_do_not_change_fitted_offsets(self):
        ref = self.reference()
        floor=pd.DataFrame({'POC_GGT':[3.,3.],'POC_Batch':[1,2],'Gamma_GT':[1.,500.]})
        config=dict(fit_slope=False,exclude_floor=True,low_weight=.3)
        expected=fit_calibration(ref,config)
        actual=fit_calibration(pd.concat([ref,floor]),config)
        pd.testing.assert_series_equal(actual.offsets,expected.offsets)
        pd.testing.assert_series_equal(actual.effective_counts,expected.effective_counts)

    def test_empty_unknown_and_missing_batches_keep_fallback(self):
        ref = self.reference()
        config=dict(exclude_floor=True,floor_mode='censored',adaptive=True)
        frame=pd.DataFrame({'POC_GGT':[np.nan,3.,6.], 'POC_Batch':[1,999,999]})
        fallback=np.array([12.,15.,20.])
        p,known,_=calibrated_prediction(fit_calibration(ref,config),frame,fallback,config)
        np.testing.assert_allclose(p,fallback);self.assertFalse(known.any())
        state=fit_calibration(ref.assign(POC_GGT=3),config)
        p,known,_=calibrated_prediction(state,frame,fallback,config)
        np.testing.assert_allclose(p,fallback);self.assertFalse(known.any())

    def test_censored_update_and_extreme_prior_are_finite(self):
        config=dict(exclude_floor=True,floor_mode='censored',adaptive=True,prior_sd=.27)
        ref=self.reference()
        frame=pd.DataFrame({'POC_GGT':[3.,3.,3.], 'POC_Batch':[1,2,3]})
        fallback=np.array([1.,20.,1000.])
        p,known,floor=calibrated_prediction(fit_calibration(ref,config),frame,fallback,config)
        self.assertTrue(np.isfinite(p).all());self.assertTrue(known.all() and floor.all())
        self.assertTrue(np.logical_and(p>=1,p<=fallback+1e-12).all())


class CrossfitIsolationTests(unittest.TestCase):
    def test_each_training_probability_excludes_its_own_labels(self):
        reference=pd.DataFrame({'ID':np.arange(30),'Smoking':np.tile([0,1],15)})
        calls=[]
        class FakeModel:
            def __init__(self, ids): self.ids=set(ids)
            def save_model(self,path): pass
        def fit(ref,name,seed):
            calls.append(set(ref.ID))
            return FakeModel(ref.ID),10
        def predict(model,ref,frame,name):
            self.assertFalse(model.ids.intersection(frame.ID))
            self.assertEqual(model.ids,set(ref.ID))
            return np.full(len(frame),.5)
        with tempfile.TemporaryDirectory() as directory:
            with patch('cross_target.fit_one',side_effect=fit),patch('cross_target.predict_one',side_effect=predict):
                oof,model=smoking_features(reference,directory,101)
            audit=json.loads((Path(directory)/'crossfit_audit.json').read_text())
        np.testing.assert_allclose(oof,.5)
        self.assertEqual(len(calls),4)
        predicted=[]
        for fold in audit:
            self.assertFalse(set(fold['fitting_ids']).intersection(fold['predicted_ids']))
            predicted+=fold['predicted_ids']
        self.assertEqual(sorted(predicted),reference.ID.tolist())


if __name__=='__main__':unittest.main()
