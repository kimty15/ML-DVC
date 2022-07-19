# Data manipulation
from seaborn import load_dataset
import numpy as np
import pandas as pd

pd.options.display.precision = 4
pd.options.mode.chained_assignment = None  

# Load data
columns = ['alive', 'class', 'embarked', 'who', 'alone', 'adult_male']
df = load_dataset('titanic').drop(columns=columns)
df['deck'] = df['deck'].astype('object')
df.to_csv("data.csv")

