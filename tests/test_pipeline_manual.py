import joblib
import pandas as pd

preprocessor = joblib.load("data/preprocessor.joblib")

new_data = pd.DataFrame(
    {
        "name": ["Eve"],
        "age": [30],
        "city": ["Pune"],
        "salary": [55000],
    }
)

processed_data = preprocessor.transform(new_data)

print(processed_data)
