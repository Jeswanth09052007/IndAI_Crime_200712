import pandas as pd

df = pd.read_csv("cleaned_crime_data.csv")

# Convert date
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Group by location (e.g., state + city or pincode if available)
crime_by_area = df.groupby(['state', 'city']).size().reset_index(name='total_crimes')

# Sort to see hotspots
crime_by_area = crime_by_area.sort_values(by='total_crimes', ascending=False)
print(crime_by_area.head(10))  # Top 10 crime-prone areas
