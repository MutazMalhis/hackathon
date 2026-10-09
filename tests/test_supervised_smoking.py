"""Exact own-label exclusion for label-adjusted urine plate statistics."""
import unittest

import numpy as np
import pandas as pd

from supervised_smoking import marker_feature


class SupervisedMarkerTests(unittest.TestCase):
    def reference(self):
        return pd.DataFrame({'ID':range(12),'Urine_Plate':np.repeat([1,2,3],4),
            'Smoking':np.tile([0,1,0,1],3),'Urine_Creatinine':100.,
            'Urine_Cotinine_ng_mL':np.exp(np.repeat([1.,2.,3.],4)+np.tile([0.,1.2,.1,1.3],3))*100})

    def test_own_label_does_not_change_own_feature(self):
        ref=self.reference();expected=marker_feature(ref,ref,True)
        for i in range(len(ref)):
            changed=ref.copy();changed.loc[i,'Smoking']=1-changed.loc[i,'Smoking']
            actual=marker_feature(changed,changed,True)
            np.testing.assert_allclose(expected.iloc[i],actual.iloc[i],atol=1e-12)

    def test_training_transform_equals_explicit_row_exclusion(self):
        ref=self.reference();actual=marker_feature(ref,ref,True)
        for i in range(len(ref)):
            expected=marker_feature(ref.drop(index=i),ref.iloc[[i]])
            np.testing.assert_allclose(actual.iloc[i],expected.iloc[0],atol=1e-12)

    def test_evaluation_labels_are_unused_and_unknown_plate_is_finite(self):
        ref=self.reference();frame=ref.iloc[:2].copy();frame['Urine_Plate']=-999
        expected=marker_feature(ref,frame)
        changed=frame.copy();changed['Smoking']=1-changed.Smoking
        actual=marker_feature(ref,changed)
        pd.testing.assert_frame_equal(expected,actual)
        self.assertTrue(np.isfinite(actual).all().all())


if __name__=='__main__':unittest.main()
