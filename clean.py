import pandas as pd

# Load merged CSV file
df = pd.read_csv("merged_crime_data.csv", low_memory=False)

# Show basic info
print("🔍 Shape of dataset:", df.shape)
print("\n🧠 Column Names:\n", df.columns)
print("\n👀 Sample Data:\n", df.head())

# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Handle missing values (fill with 'Unknown')
df.fillna("Unknown", inplace=True)

# Normalize column names (lowercase, no spaces)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Save cleaned data
df.to_csv("cleaned_crime_data.csv", index=False)
print("\n✅ Cleaned data saved to 'cleaned_crime_data.csv'")
