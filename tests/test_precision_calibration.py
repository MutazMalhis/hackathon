import unittest

import numpy as np
import pandas as pd

from calibration_precision_push import fit_interval_calibration, interval_predictions, weighted_calibration


class PrecisionCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(9301)
        gamma=np.exp(rng.uniform(np.log(4),np.log(150),size=240))
        batch=np.repeat(['A','B'],120)
        offset=np.where(batch=='A',-.25,.35)
        reading=np.maximum(3,np.round(np.exp(np.log(gamma)+offset+rng.normal(0,.06,240))))
        cls.train=pd.DataFrame(dict(POC_Batch=batch,POC_GGT=reading,Gamma_GT=gamma))
        cls.model=fit_interval_calibration(cls.train)

    def test_synthetic_batch_offset_recovery(self):
        self.assertAlmostEqual(self.model['offset']['A'],-.25,delta=.04)
        self.assertAlmostEqual(self.model['offset']['B'],.35,delta=.04)

    def test_held_labels_ignored_and_unknown_readings_keep_prior(self):
        frame=pd.DataFrame(dict(POC_Batch=['A','NEW','B','A'],POC_GGT=[3,10,np.nan,1000],Gamma_GT=[1,2,3,4]))
        prior=np.log1p([1000.,20.,40.,1.])
        _,first=interval_predictions(self.model,frame,prior)
        frame.Gamma_GT=frame.Gamma_GT*1000
        _,second=interval_predictions(self.model,frame,prior)
        np.testing.assert_array_equal(first,second)
        np.testing.assert_array_equal(first[[1,2]],prior[[1,2]])
        self.assertTrue(np.isfinite(first).all())

    def test_floor_labels_do_not_affect_corrected_weighted_slope(self):
        frame=self.train.iloc[:10].copy()
        train=self.train.copy();train.loc[0,'POC_GGT']=3
        expected,beta=weighted_calibration(train,frame)
        train.loc[0,'Gamma_GT']=1e6
        result,other=weighted_calibration(train,frame)
        np.testing.assert_array_equal(expected,result)
        self.assertEqual(beta,other)

    def test_unavailable_calibration_uses_prior(self):
        frame=pd.DataFrame(dict(POC_Batch=['A'],POC_GGT=[3.],Gamma_GT=[10.]))
        model=fit_interval_calibration(frame)
        _,actual=interval_predictions(model,frame,np.log1p([12.]))
        np.testing.assert_array_equal(actual,np.log1p([12.]))


if __name__=='__main__':unittest.main()
