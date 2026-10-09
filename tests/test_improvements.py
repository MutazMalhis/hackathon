"""Run after downloading the competition CSVs: python -m unittest discover -s tests."""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from experiments import batch_correct
from improvements import engineered_features, development_data


@unittest.skipUnless(Path('data/train.csv').exists(), 'Competition data must be downloaded first.')
class FeatureIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = pd.read_csv('data/train.csv')
        cls.reference = data.iloc[:300].copy()
        cls.frame = data.iloc[300:330].copy()

    def test_targets_and_ids_are_excluded(self):
        for variant in ['urine', 'enriched', 'clinical']:
            x, _ = engineered_features(self.reference, self.frame, variant)
            self.assertFalse({'ID', 'Smoking', 'Gamma_GT'}.intersection(x))
        self.assertFalse({'POC_GGT', 'POC_Batch'}.intersection(x))

    def test_features_do_not_depend_on_evaluation_targets(self):
        expected, _ = engineered_features(self.reference, self.frame, 'enriched')
        modified = self.frame.copy()
        modified['Smoking'] = 1-modified.Smoking
        modified['Gamma_GT'] = 1000
        actual, _ = engineered_features(self.reference, modified, 'enriched')
        assert_frame_equal(expected, actual)

    def test_plate_statistics_use_only_reference_features(self):
        expected, _ = engineered_features(self.reference, self.frame, 'enriched')
        modified = self.reference.copy()
        modified['Smoking'] = 1-modified.Smoking
        modified['Gamma_GT'] = 1000
        actual, _ = engineered_features(modified, self.frame, 'enriched')
        assert_frame_equal(expected, actual)
        # Other evaluation rows cannot affect a given row's normalization.
        modified = self.frame.copy()
        modified.loc[modified.index[1:], 'Urine_Cotinine_ng_mL'] = 1e9
        actual, _ = engineered_features(self.reference, modified, 'enriched')
        assert_frame_equal(expected.iloc[:1], actual.iloc[:1])

    def test_unseen_plate_and_missing_cartridge_fallback(self):
        frame = self.frame.copy()
        frame['Urine_Plate'] = -999
        frame['POC_Batch'] = -999
        x, cats = engineered_features(self.reference, frame, 'clinical')
        self.assertFalse(np.isinf(x.drop(columns=cats).to_numpy()).any())
        fallback = np.linspace(10, 30, len(frame))
        prediction, covered = batch_correct(self.reference, frame, fallback, weight=.9)
        self.assertFalse(covered.any())
        np.testing.assert_allclose(prediction, fallback)
        frame['POC_GGT'] = np.nan
        prediction, covered = batch_correct(self.reference, frame, fallback, weight=.9)
        self.assertFalse(covered.any())
        np.testing.assert_allclose(prediction, fallback)

    @unittest.skipUnless(Path('outputs/plan_v1/split_ids.csv').exists(), 'Run experiments.py first.')
    def test_development_selection_excludes_previous_holdout(self):
        split = pd.read_csv('outputs/plan_v1/split_ids.csv')
        dev = development_data()
        held_ids = set(split.loc[split.partition.eq('holdout'), 'ID'])
        self.assertFalse(set(dev.ID).intersection(held_ids))
        self.assertEqual(len(dev), int(split.partition.eq('development').sum()))


if __name__ == '__main__':
    unittest.main()
