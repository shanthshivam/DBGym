from typing import List
import os
import pandas as pd

def split_csv_by_time(path: str, 
                      target_csv: str, 
                      time_col: str, 
                      ratio: List =[0.7,0.2,0.1]
                      ):
    '''
    This function is used to divide CSVs into train, val, test chronologically.

    Steps:
        1. Determine time threshold according to one time column.
        2. Divide all CSVs according to given time threshold.
    
    Args:
        path: str 
            The path of the dataset you want to divide.
        target_csv: str
            This argument should be in the form of "csv_name.csv", e.g. "boy.csv",
            specifying on which csv we perform the split. 
            And we assume that all CSVs with time column is in ascending order.
        time_col: str
            This argument should be in the form of "col", e.g. "marry_time",
            specifying on which time column we perform the split. 
        ratio: List =[0.7,0.2,0.1] 
            The ratio of the split.

    Returns:
        None.
    '''
    names = ["train","val","test"]

    # Specify the directory path you want to check
    directory_path = os.path.join(path, "train")

    # Check if the directory exists
    if os.path.exists(directory_path):
        print(f"The directory '{directory_path}' exists.")
        return

    target_csv = target_csv + ".csv"
    # Step 1
    time_thresholds = list()
    files = os.listdir(path)
    for filename in files:
        # print(filename,target_csv)
        if filename == target_csv:
            df = pd.read_csv(os.path.join(path, target_csv))
            # Convert the specified time column to datetime format 
            if time_col not in df.columns.tolist():
                raise ValueError("The column does not exist.")
            df[time_col] = pd.to_datetime(df[time_col])
            tot = len(df[time_col])
            proportion = 0
            for i in ratio:
                proportion += i
                if abs(proportion-1) > 1e-5:
                    index = int(proportion * tot)
                else:
                    index = tot
                time_thresholds.append(df[time_col][index-1])
            # print(time_thresholds)
    
    # Step 2
    for filename in files:
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(path, filename))

            # Convert the columns ending with "_time" to datetime
            time_columns = [col for col in df.columns if col.endswith('_time')]
            if len(time_columns) != 0:
                for col in time_columns:
                    df[col] = pd.to_datetime(df[col])
                for time_threshold, name in zip(time_thresholds, names):
                    # Filter rows based on the given datetime and condition
                    filtered_data = df[df[time_columns].apply(lambda col: (col <= time_threshold).all(), axis=1)]
                    file_path = os.path.join(path, name, filename)
                    # Extract the directory path from the file path
                    directory = os.path.dirname(file_path)

                    # Create the parent directories if they don't exist
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    filtered_data.to_csv(file_path, index=False)
            else: # It has not.
                for name in names:
                    file_path = os.path.join(path, name, filename)
                    # Extract the directory path from the file path
                    directory = os.path.dirname(file_path)

                    # Create the parent directories if they don't exist
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    df.to_csv(file_path, index=False)

    

if __name__ == "__main__":
    path = "./Datasets/tiny_example"
    target_csv = "couple.csv"
    time_col = "marry_time"
    ratio = [0.8,0.1,0.1]
    split_csv_by_time(path, target_csv, time_col, ratio)


