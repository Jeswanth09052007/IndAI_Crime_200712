import os
import pandas as pd
import glob

folder_path = r"C:\Users\Jeswa\OneDrive\Desktop\Personal\IIITS\Hackathons\ThunderDome\Crime Data\Historical Crime Data"
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

dataframes = []

for file in csv_files:
    try:
        df = pd.read_csv(file, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        print(f"⚠️ UTF-8 failed for {file}, trying latin1...")
        df = pd.read_csv(file, encoding='latin1', low_memory=False)
    dataframes.append(df)

merged_df = pd.concat(dataframes, ignore_index=True)

# Save to a single CSV
merged_df.to_csv("merged_crime_data.csv", index=False)
print("✅ All files merged and saved to 'merged_crime_data.csv'")
