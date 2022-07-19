from sklearn.metrics import accuracy_score
from seaborn import load_dataset
import numpy as np
import pandas as pd
from functions import calculate_roc_auc
pd.options.display.precision = 4
pd.options.mode.chained_assignment = None  
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import FeatureExtractor, Imputer, CardinalityReducer, Encoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
SEED = 42
TARGET = 'survived'
FEATURES = df.columns.drop(TARGET)
NUMERICAL = df[FEATURES].select_dtypes('number').columns
print(f"Numerical features: {', '.join(NUMERICAL)}")
CATEGORICAL = pd.Index(np.setdiff1d(FEATURES, NUMERICAL))
print(f"Categorical features: {', '.join(CATEGORICAL)}\n")
pipe = Pipeline([
    ('feature_extractor', FeatureExtractor()), 
    ('cat_imputer', Imputer(CATEGORICAL)), 
    ('cardinality_reducer', CardinalityReducer(CATEGORICAL, threshold=0.1)),
    ('encoder', Encoder(CATEGORICAL)),
    ('num_imputer', Imputer(NUMERICAL, method='mean')), 
    ('feature_selector', RFE(LogisticRegression(random_state=SEED, max_iter=500), n_features_to_select=8)), 
    ('model', DecisionTreeRegressor(random_state=SEED, max_features=auto))
])
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=TARGET), df[TARGET], 
                                                    test_size=.2, random_state=SEED, 
                                                    stratify=df[TARGET])
                                                    
pipe.fit(X_train, y_train)
accuracy_score = pipe.score(X_test, y_test)

with open("metrics.txt", "w") as outfile:
    outfile.write("Accuracy: " + str(accuracy_score) + "\n")


disp = plot_confusion_matrix(pipe, X_test, y_test, normalize="true", cmap=plt.cm.Blues)
plt.savefig("plot.png")
