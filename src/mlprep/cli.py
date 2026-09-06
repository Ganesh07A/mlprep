import typer
import pandas as pd
app = typer.Typer()


@app.command()
def inspect(file_path:str):
    df = pd.read_csv(file_path)

    print(f"Rows:{df.shape[0]}")
    print(f"Columns:{df.shape[1]}")
    print(f"\n Missing values per column\n")
    print(df.isnull().sum())
    print("\n Data Types :\n")
    print(df.dtypes)
    print("\n Sample Data :\n")
    print(df.head())

if __name__ == "__main__":
    app()