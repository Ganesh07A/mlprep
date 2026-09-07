import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def create_encoder() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


# tesing purpose only
data = pd.DataFrame({
    "city": ["Delhi", "Mumbai", "Delhi"]
})

encoder = OneHotEncoder(sparse_output=False)

encoded = encoder.fit_transform(data)

print(encoded)
print(encoder.get_feature_names_out())