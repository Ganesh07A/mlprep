from mlprep.profiler import profile_dataset
import typer
import pandas as pd
app = typer.Typer()


@app.command()
def inspect(file_path:str):
    df = pd.read_csv(file_path)
    profile = profile_dataset(df)

    print("\nMissing values:")
    print(profile.missing_values)

    print("\nData Types:")
    print(profile.data_types)

    print("\nCategorical Values:")
    print(profile.categorical_values)

    print("\nNumerical Values:")
    print(profile.numerical_values)

    print("\nSample Data:")
    print(profile.sample_data)

if __name__ == "__main__":
    app()