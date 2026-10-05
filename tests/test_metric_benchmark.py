"""Metric evidence checks run without loading BERT or any saved model artifact."""
import unittest

import numpy as np
import pandas as pd

from ml.training.benchmark_metrics import confusion_metrics, select_threshold, temporal_partitions


class BenchmarkEvidenceTests(unittest.TestCase):
    def test_confusion_counts_include_sample_size_and_positive_support(self):
        result=confusion_metrics([0,0,1,1],[0,1,1,0],[.1,.8,.9,.2])
        self.assertEqual((result['tp'],result['fp'],result['fn'],result['tn']),(1,1,1,1))
        self.assertEqual(result['records'],4);self.assertEqual(result['positive_records'],2)
        self.assertEqual(result['f1'],.5)

    def test_single_class_metrics_do_not_invent_roc_auc(self):
        result=confusion_metrics([0,0],[0,0],[.1,.2])
        self.assertIsNone(result['roc_auc']);self.assertEqual(result['accuracy'],1)
        self.assertEqual(result['f1'],0)

    def test_threshold_tuning_uses_validation_not_test_labels(self):
        threshold=select_threshold([0,0,1,1],[.1,.2,.8,.9])
        self.assertEqual(threshold,.8)
        low=confusion_metrics([0,1],[0,1]);high=confusion_metrics([1,0],[0,1])
        self.assertNotEqual(low['f1'],high['f1'])
        self.assertEqual(threshold,select_threshold([0,0,1,1],[.1,.2,.8,.9]))

    def test_temporal_partitions_have_disjoint_sample_identifiers(self):
        frame=pd.DataFrame({'_source_row':np.arange(100)})
        train,validation,test=temporal_partitions(frame)
        self.assertEqual((len(train),len(validation),len(test)),(60,20,20))
        self.assertEqual(test._source_row.min(),80)
        self.assertFalse(set(train._source_row)&set(test._source_row))

    def test_nonfinite_scores_and_nonbinary_labels_fail(self):
        with self.assertRaises(ValueError):select_threshold([0,1],[.1,np.nan])
        with self.assertRaises(ValueError):confusion_metrics([0,2],[0,1])
        with self.assertRaises(ValueError):confusion_metrics([],[])


if __name__=='__main__':unittest.main()
