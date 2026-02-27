import numpy as np
import pandas as pd
import sklearn as sk

df = pd.read_csv("creditcard.csv")
print(df.head())

print(df["Class"].value_counts())


majority = df[df["Class"] == 0]
minority = df[df["Class"] == 1]

print("Majority shape:", majority.shape)
print("Minority shape:", minority.shape)

from sklearn.utils import resample

#under sampling 

rs = resample(
    majority,
    replace=False,
    n_samples=len(minority)
)

balanced_df = pd.concat([rs, minority])

print(balanced_df["Class"].value_counts())


# over sampling     

minority_over = resample(
    minority,
    replace=True,                 
    n_samples=len(majority),      
    random_state=42
)

over_balanced_df = pd.concat([majority, minority_over])

print("\nAfter Over Sampling:")
print(over_balanced_df["Class"].value_counts())    
