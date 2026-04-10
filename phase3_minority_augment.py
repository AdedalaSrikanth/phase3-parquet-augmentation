from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import itertools


script_dir = Path(__file__).resolve().parent

# input and output file names
input_file = "input.parquet"
output_file = "augmented_minority_only.parquet"

# column names
target_column = "target"
group_column = "group_id"
source_row_id_column = "source_row_id"
is_augmented_column = "is_augmented"
augmentation_round_column = "augmentation_round"
randomization_mode_column = "randomization_mode"

# how much noise to add - keeping it small so the data doesnt change too much
noise_std = 0.01
random_seed = 42

# set this to group_based so we can track which samples came from which original row
# this is important for cross validation later so augmented copies dont leak into the test set
randomization_mode = "group_based"


def get_input_path(filename):
    file_path = script_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(
            f"could not find '{filename}', make sure it is in the same folder as this script: {script_dir}"
        )
    return file_path


def get_output_path(filename):
    return script_dir / filename


def load_parquet_file(filename):
    file_path = get_input_path(filename)
    return pd.read_parquet(file_path)


def ensure_source_row_id(df):
    # add a source_row_id column if it doesnt already exist
    # this lets us track which original row each augmented row came from
    df = df.copy()
    if source_row_id_column not in df.columns:
        df[source_row_id_column] = np.arange(len(df))
    return df


def ensure_group_id(df, mode):
    df = df.copy()

    if mode == "group_based":
        # if group_id isnt already there we assign one per row
        # augmented copies will inherit this id from their source row
        # so we can keep them together when splitting data for cross validation
        if group_column not in df.columns:
            df[group_column] = np.arange(len(df))
    elif mode == "normal":
        # in normal mode we dont need group ids so just drop it if it exists
        if group_column in df.columns:
            df = df.drop(columns=[group_column])

    return df


def prepare_original_rows(df, mode):
    # mark all the original rows before we add the augmented ones
    df = df.copy()
    df[is_augmented_column] = 0
    df[augmentation_round_column] = 0
    df[randomization_mode_column] = mode
    return df


def get_numeric_feature_columns(df):
    # we only want to add noise to the actual feature columns
    # not the metadata columns we added like group_id, source_row_id etc
    skip_columns = {
        target_column,
        source_row_id_column,
        is_augmented_column,
        augmentation_round_column,
        randomization_mode_column,
    }

    if group_column in df.columns:
        skip_columns.add(group_column)

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_columns if col not in skip_columns]


def augment_row(row, numeric_columns, rng, round_number, mode):
    # make a copy of the row and add a tiny bit of random noise to each feature
    # we chose gaussian noise over SMOTE because SMOTE needs two minority samples
    # to interpolate between, which makes tracking source IDs more complicated
    # this way every augmented row has exactly one source which is simpler
    new_row = row.copy()

    for col in numeric_columns:
        value = row[col]
        if pd.notna(value):
            noise = rng.normal(loc=0.0, scale=noise_std)
            new_row[col] = value + noise

    new_row[is_augmented_column] = 1
    new_row[augmentation_round_column] = round_number
    new_row[randomization_mode_column] = mode
    return new_row


def generate_minority_augmented_data(df, mode):
    numeric_columns = get_numeric_feature_columns(df)

    if not numeric_columns:
        raise ValueError("no numeric columns found to augment")

    rng = np.random.default_rng(random_seed)

    # figure out which class has fewer samples
    class_counts = df[target_column].value_counts()
    minority_value = class_counts.idxmin()
    majority_count = int(class_counts.max())

    minority_rows = df[df[target_column] == minority_value].copy()
    minority_count = len(minority_rows)

    if minority_count == 0:
        raise ValueError("minority class is empty, nothing to augment")

    # how many extra rows do we need to match the majority class size
    needed_rows = majority_count - minority_count

    if needed_rows <= 0:
        print("classes are already balanced, no augmentation needed")
        return pd.DataFrame(columns=df.columns)

    print(f"minority class has {minority_count} rows, majority has {majority_count}")
    print(f"need to generate {needed_rows} augmented rows")

    augmented_rows = []

    # cycle through the minority rows so each source sample gets roughly the same
    # number of augmented copies - this is better than doing all copies for one row
    # then all for the next because the last few rows would get cut off otherwise
    minority_iter = itertools.cycle(minority_rows.iterrows())

    for round_number, (_, row) in enumerate(itertools.islice(minority_iter, needed_rows), 1):
        augmented_rows.append(
            augment_row(
                row=row,
                numeric_columns=numeric_columns,
                rng=rng,
                round_number=round_number,
                mode=mode,
            )
        )

    return pd.DataFrame(augmented_rows, columns=df.columns)


def save_parquet_file(df, filename):
    output_path = get_output_path(filename)
    df.to_parquet(output_path, index=False)
    print(f"saved output to: {output_path.name}")


def augment_minority_only_file(input_filename=input_file, output_filename=output_file, mode=randomization_mode):
    if mode not in {"normal", "group_based"}:
        raise ValueError("mode must be either 'normal' or 'group_based'")

    df = load_parquet_file(input_filename)
    df = ensure_source_row_id(df)
    df = ensure_group_id(df, mode)
    df = prepare_original_rows(df, mode)

    augmented_df = generate_minority_augmented_data(df, mode)

    # combine original rows with the augmented minority rows
    final_df = pd.concat([df, augmented_df], ignore_index=True)

    save_parquet_file(final_df, output_filename)
    return final_df


def main():
    print("starting augmentation")
    print(f"script folder: {script_dir}")
    print(f"randomization mode: {randomization_mode}")

    final_df = augment_minority_only_file()

    original_count = (final_df[is_augmented_column] == 0).sum()
    augmented_count = (final_df[is_augmented_column] == 1).sum()

    print(f"original rows: {original_count}")
    print(f"augmented rows added: {augmented_count}")
    print(f"total rows in output: {len(final_df)}")
    print("done")


if __name__ == "__main__":
    main()