import pandas as pd

df = pd.read_csv("cleaned_crime_data.csv")  # Use your merged + cleaned dataset

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
