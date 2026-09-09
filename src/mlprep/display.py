import pandas as pd
from rich.console import Console
from rich.table import Table

from mlprep.profiler import profile_dataset


def display_profile(df: pd.DataFrame):
    profile = profile_dataset(df)
    console = Console()
    table = Table(title="MISSING VALUES")

    table.add_column("Columns")
    table.add_column("Missing")
    table.add_column("Percentage")

    for column in profile.missing_count.index:
        table.add_row(
            column,
            str(profile.missing_count[column]),
            f"{profile.missing_percentage[column]:.1f}%",
        )

    console.print(table)

    # data_type table
    types_table = Table(title="Data Types")

    types_table.add_column("Column")
    types_table.add_column("DataType")

    for column in profile.data_types.index:
        if column in profile.numerical_values:
            column_type = "numerical"
        elif column in profile.categorical_values:
            column_type = "Categorical"
        else:
            column_type = str(profile.data_types[column])

        types_table.add_row(column, column_type)

    console.print(types_table)
