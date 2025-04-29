import pandas as pd
import numpy as np
import pickle
import os

directory_path = 'D:/Python/Hydraulic rig Dataset'

# 2. Helper function to read a file robustly
def read_file_robust(filepath, sep=r"\s+"):
    """
    Attempt to read a file using UTF-8; if that fails, try ISO-8859-1.
    Skips lines with inconsistent column counts.
    """
    try:
        return pd.read_csv(
            filepath,
            sep=sep,
            header=None,
            engine="python",
            encoding="utf-8",
            on_bad_lines="skip"
        )
    except UnicodeDecodeError:
        return pd.read_csv(
            filepath,
            sep=sep,
            header=None,
            engine="python",
            encoding="ISO-8859-1",
            on_bad_lines="skip"
        )

# 3. Identify sensor files and the profile file
sensor_files = []
profile_file = None

for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    
    # Skip directories
    if not os.path.isfile(file_path):
        continue
    
    # Skip files that are documentation or description files
    lower_filename = filename.lower()
    if "documentation" in lower_filename or "description" in lower_filename:
        continue
    
    # If it's the profile file, store it separately
    if lower_filename.startswith("profile") and lower_filename.endswith(".txt"):
        profile_file = file_path
    # Otherwise, assume it's a sensor data file if it ends with .txt
    elif lower_filename.endswith(".txt"):
        sensor_files.append(file_path)

# 4. Read sensor files into DataFrames with encoded column names
sensor_dfs = []
for sf in sensor_files:
    base_name = os.path.splitext(os.path.basename(sf))[0]  # e.g. "PS5"
    print(f"Reading sensor file: {os.path.basename(sf)}")
    
    # Read sensor file data
    df_sensor = read_file_robust(sf, sep=r"\s+")
    
    # Rename columns to reflect the file name + index (e.g., PS5_1, PS5_2, ...)
    df_sensor.columns = [f"{base_name}_{i+1}" for i in range(df_sensor.shape[1])]
    sensor_dfs.append(df_sensor)

# 5. Combine all sensor DataFrames column-wise (assuming same number of rows)
if sensor_dfs:
    sensor_data_df = pd.concat(sensor_dfs, axis=1)
    print(f"Combined sensor data shape: {sensor_data_df.shape}")
    if sensor_data_df.shape[1] != 43680:
        print(f"Warning: The sensor data has {sensor_data_df.shape[1]} features; expected 43680.")
else:
    sensor_data_df = pd.DataFrame()

# 6. Read the profile file (target dataset) separately
if profile_file is not None:
    print(f"Reading profile file: {os.path.basename(profile_file)}")
    profile_df = read_file_robust(profile_file, sep=r"\s+")
    
    # Expected column names for profile data (if exactly 4 columns)
    expected_profile_cols = ["cooler_condition", "valve_condition", "pump_leakage", "accumulator_condition"]
    if profile_df.shape[1] == 4:
        profile_df.columns = expected_profile_cols
    else:
        profile_df.columns = [f"profile_{i+1}" for i in range(profile_df.shape[1])]
    
    print(f"Profile data shape: {profile_df.shape}")
else:
    profile_df = pd.DataFrame()

# 7. Do NOT merge sensor_data_df with profile_df.
print("Sensor Data Preview:")
print(sensor_data_df.head())
print(sensor_data_df.head())
print(sensor_data_df.shape[0])
print(sensor_data_df.shape[1])


profile_df.drop(["profile_1",  "profile_2",  "profile_3",  "profile_4",], axis=1, inplace=True)
profile_df.rename(columns={"profile_5": "Target"}, inplace=True)


print("Profile Data Preview:")
print(profile_df.head())
print(profile_df.head())
print(profile_df.shape[0])
print(profile_df.shape[1])



# 8. Pickle the DataFrames separately
sensor_pickle_path = r"D:\Python\Hydraulic Rig Dataset\Data\sensor_data_df.pkl"
profile_pickle_path = r"D:\Python\Hydraulic Rig Dataset\Data\profile_df.pkl"

sensor_data_df.to_pickle(sensor_pickle_path)
profile_df.to_pickle(profile_pickle_path)

print(f"Sensor data pickled to: {sensor_pickle_path}")
print(f"Profile data pickled to: {profile_pickle_path}")
