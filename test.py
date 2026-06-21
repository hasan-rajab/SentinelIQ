# Check if the test set has totally non-overlapping bytes_out_in_ratio between classes
import pandas as pd
df = pd.read_csv("data/processed/network_test.csv")  # or wherever your test set's engineered features are
from ml.features.network_features import add_network_features
df = add_network_features(df)

normal = df[df['is_anomaly']==0]['bytes_out_in_ratio']
anomaly = df[df['is_anomaly']==1]['bytes_out_in_ratio']
print("Normal range:", normal.min(), normal.max())
print("Anomaly range:", anomaly.min(), anomaly.max())