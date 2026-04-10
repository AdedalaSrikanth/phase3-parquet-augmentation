from pathlib import Path
import pandas as pd

script_dir = Path(__file__).resolve().parent

input_csv = "gallstone.csv"
output_parquet = "input.parquet"

target_source_column = "Gallstone Status"


def main():
    csv_path = script_dir / input_csv

    if not csv_path.exists():
        raise FileNotFoundError(f"file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    print("original columns:")
    print(df.columns.tolist())

    if target_source_column not in df.columns:
        raise ValueError(
            f"target source column '{target_source_column}' not found in csv. "
            f"choose one from: {df.columns.tolist()}"
        )

    df = df.rename(columns={target_source_column: "target"})

    for column in df.columns:
        if df[column].dtype == "object" and column != "target":
            df[column] = pd.factorize(df[column])[0]

    if df["target"].dtype == "object":
        df["target"] = pd.factorize(df["target"])[0]

    df.to_parquet(script_dir / output_parquet, index=False)

    print("saved: input.parquet")
    print(df.head())
    print(df.dtypes)


if __name__ == "__main__":
    main()