"""Held-out record-level benchmark on the committed synthetic metric stream.

Fits a fresh Isolation Forest on the first 60%, calibrates on the next 20%,
then scores the final 20% once. No simulator label enters model features.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.models.isolation_forest import SentinelIsolationForest

ALLOWED_FEATURES={'cpu_percent','mem_percent','disk_read_mbps','disk_write_mbps',
                  'net_in_mbps','net_out_mbps','open_connections','process_count'}


def load_records(path):
    records=[]
    with Path(path).open() as handle:
        for line_number,line in enumerate(handle,1):
            if not line.strip(): continue
            row=json.loads(line)
            if type(row.get('is_anomaly')) is not bool:
                raise ValueError(f'Binary simulator label required on line {line_number}')
            timestamp=datetime.fromisoformat(row['timestamp'].replace('Z','+00:00'))
            if timestamp.tzinfo is None: timestamp=timestamp.replace(tzinfo=timezone.utc)
            row['_timestamp']=timestamp.astimezone(timezone.utc)
            row['_source_row']=line_number
            records.append(row)
    if len(records)<30: raise ValueError('At least 30 observations required for three partitions')
    return pd.DataFrame(records).sort_values(['_timestamp','_source_row']).reset_index(drop=True)


def temporal_partitions(frame,train_fraction=.6,validation_fraction=.2):
    if not 0<train_fraction<1 or not 0<validation_fraction<1 or train_fraction+validation_fraction>=1:
        raise ValueError('Invalid partition fractions')
    first=int(len(frame)*train_fraction);second=int(len(frame)*(train_fraction+validation_fraction))
    train=frame.iloc[:first].copy();validation=frame.iloc[first:second].copy();test=frame.iloc[second:].copy()
    if min(len(train),len(validation),len(test))<1: raise ValueError('Empty partition')
    if set(train._source_row)&set(test._source_row) or set(validation._source_row)&set(test._source_row):
        raise ValueError('Overlapping train/validation/test row IDs')
    return train,validation,test


def confusion_metrics(labels,predictions,scores=None):
    labels=np.asarray(labels); predictions=np.asarray(predictions)
    if labels.shape!=predictions.shape or labels.ndim!=1 or not len(labels):
        raise ValueError('Non-empty aligned binary vectors required')
    if not np.isin(labels,[0,1]).all() or not np.isin(predictions,[0,1]).all():
        raise ValueError('Binary labels and predictions required')
    tp=int(((labels==1)&(predictions==1)).sum());fp=int(((labels==0)&(predictions==1)).sum())
    fn=int(((labels==1)&(predictions==0)).sum());tn=int(((labels==0)&(predictions==0)).sum())
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
    result={'records':len(labels),'positive_records':tp+fn,'negative_records':tn+fp,
            'predicted_positive_records':tp+fp,'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'precision':precision,'recall':recall,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'false_positive_rate':fp/(fp+tn) if fp+tn else None,'accuracy':(tp+tn)/len(labels)}
    if scores is not None:
        scores=np.asarray(scores,dtype=float)
        if scores.shape!=labels.shape or not np.isfinite(scores).all(): raise ValueError('Invalid score vector')
        both=len(np.unique(labels))==2
        result.update(roc_auc=float(roc_auc_score(labels,scores)) if both else None,
                      average_precision=float(average_precision_score(labels,scores)) if both else None)
    return result


def select_threshold(validation_labels,validation_scores):
    scores=np.asarray(validation_scores,dtype=float)
    if not len(scores) or not np.isfinite(scores).all(): raise ValueError('Invalid validation scores')
    candidates=np.unique(np.r_[scores,np.nextafter(scores.max(),math.inf)])
    choices=[]
    for threshold in candidates:
        metrics=confusion_metrics(validation_labels,scores>=threshold)
        choices.append((metrics['f1'],metrics['precision'],float(threshold)))
    return max(choices)[2]


def benchmark(path='data/simulated/metrics.jsonl',config_path='configs/model_config.yaml'):
    path=Path(path);frame=load_records(path);train,validation,test=temporal_partitions(frame)
    config=yaml.safe_load(Path(config_path).read_text())['isolation_forest']
    features=config['features']
    if not features or len(features)!=len(set(features)) or set(features)-ALLOWED_FEATURES:
        raise ValueError('Features must be unique observed telemetry columns; labels and anomaly types are forbidden')
    for part in [train,validation,test]:
        if not np.isfinite(part[features].to_numpy(float)).all(): raise ValueError('Invalid metric feature')
    model=SentinelIsolationForest(config,features)
    with contextlib.redirect_stdout(io.StringIO()): model.fit(train[features])
    val_scores=model.score(validation[features])
    threshold=select_threshold(validation.is_anomaly.to_numpy(int),val_scores)
    test_scores=model.score(test[features])
    predictions=(test_scores>=threshold).astype(int)
    results=confusion_metrics(test.is_anomaly.to_numpy(int),predictions,test_scores)
    per_host=[]
    for host,indices in test.groupby('host').groups.items():
        positions=test.index.get_indexer(indices)
        per_host.append({'host':host,**confusion_metrics(test.loc[indices,'is_anomaly'].to_numpy(int),predictions[positions])})
    per_type=[]
    for kind,indices in test[test.is_anomaly].groupby('anomaly_type').groups.items():
        positions=test.index.get_indexer(indices);caught=int(predictions[positions].sum())
        per_type.append({'type':kind,'positive_records':len(indices),'detected_records':caught,'recall':caught/len(indices)})
    partitions={name:{'records':len(part),'positive_records':int(part.is_anomaly.sum()),
                     'first_source_row':int(part._source_row.iloc[0]),'last_source_row':int(part._source_row.iloc[-1]),
                     'start_utc':part._timestamp.iloc[0].isoformat(),'end_utc':part._timestamp.iloc[-1].isoformat()}
                for name,part in [('train',train),('validation',validation),('test',test)]}
    report={'provenance':'committed synthetic metric stream; record-level labels, not real security incidents',
            'data_path':str(path),'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'records_total':len(frame),'partitions':partitions,'features':features,'model':'Isolation Forest',
            'threshold':threshold,'threshold_selection':'max validation F1, then precision, then higher threshold; frozen for test',
            'test':results,'always_normal_baseline':confusion_metrics(test.is_anomaly.to_numpy(int),np.zeros(len(test),int)),
            'by_host':per_host,'by_anomaly_type':per_type,
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__},
            'notes':['Training sees observed features only; validation labels choose threshold; test labels only score results.',
                     'The short simulated stream cannot establish real incident accuracy or production generalization.',
                     'This standalone metric benchmark does not evaluate BERT, autoencoder or the production ensemble.',
                     'Host/type segmentation diagnoses missed records; it does not establish causal incident root causes.']}
    predictions_frame=pd.DataFrame({'source_row':test._source_row.to_numpy(),'label':test.is_anomaly.to_numpy(int),
                                    'prediction':predictions,'score':test_scores})
    return report,predictions_frame


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',default='data/simulated/metrics.jsonl')
    parser.add_argument('--config',default='configs/model_config.yaml');parser.add_argument('--outdir',default='reports/metric_benchmark')
    args=parser.parse_args();report,predictions=benchmark(args.data,args.config)
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    predictions.to_csv(out/'test_predictions.csv',index=False)
    print(json.dumps({'provenance':report['provenance'],'partitions':report['partitions'],'test':report['test']},indent=2))


if __name__=='__main__': main()
