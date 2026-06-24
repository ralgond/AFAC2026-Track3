import pandas as pd

sub = pd.read_csv("../data/A1-Cls/sample_submission.csv")
labels = [0] * len(sub)

sub['label'] = labels

sub.to_csv("A1.csv", index=False)