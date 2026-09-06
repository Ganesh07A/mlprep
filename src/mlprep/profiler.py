import pandas as pd
from mlprep.models import datasetProfile

def profile_dataset(df: pd.DataFrame) -> datasetProfile:

    # classification of column values 
    categorical_values = df.select_dtypes(include="object").columns.tolist()
    numerical_values = df.select_dtypes(include="number").columns.tolist()
    
    return datasetProfile(
        missing_values=df.isnull().sum(),
        data_types= df.dtypes,
        sample_data= df.head(),
        categorical_values= categorical_values,
        numerical_values= numerical_values,
    )