import os

import pandas as pd


def main(path_jsons: str = "output/json_benchmark"):
    """Find the json files that are not valid and deletes them.

    First, you must have run:
    scripts/realworld/results.py -> For the json files
    scripts/realworld/figures.py -> For the csv files

    Parameters
    ----------
    path_jsons : str
        Path to the folder containing the json files and the dataframe_invalid.csv file.
        Default is "output/json_benchmark".
    """
    # Locate the dataframe_invalid.csv file
    path_invalid_csv = os.path.join(path_jsons, "dataframe_invalid.csv")
    if not os.path.exists(path_invalid_csv):
        print(
            f"File {path_invalid_csv} not found. "
            "Please run scripts/realworld/figures.py first."
        )
        return
    # Read the invalid dataframe
    df_invalid = pd.read_csv(path_invalid_csv)
    # Loop through the invalid json files and delete them
    for _, row in df_invalid.iterrows():
        instance_name = row["instance_name"].replace("-", "")
        date_start = row["date_start"]
        vel_ship = row["vel_ship"]
        # Name is instance + "YYMMDD" + vel
        date_str = date_start.replace("-", "")[2:]  # Convert to YYMMDD
        json_filename = f"{instance_name}_{date_str}_{vel_ship}.json"
        json_path = os.path.join(path_jsons, json_filename)
        if os.path.exists(json_path):
            os.remove(json_path)
            print(f"Deleted invalid json file: {json_path}")
        else:
            print(f"Json file not found (already deleted?): {json_path}")


if __name__ == "__main__":
    main()
