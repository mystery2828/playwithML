import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

filename = 'pulse.xlsx'
dataframe = pd.read_csv(filename)

