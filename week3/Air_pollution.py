import pandas as pd

# Load the data into a pandas DataFrame
df = pd.read_csv('air-pollution.csv')

# Create a filter based on the country
country_filter = df['Entity'] == 'United States'

# Find the median, mean, and standard deviation for the respective columns
df['NOx_median'] = df.loc[country_filter, 'Nitrogen oxide (NOx)'].median()
df['NOx_mean'] = df.loc[country_filter, 'Nitrogen oxide (NOx)'].mean()
df['NOx_std'] = df.loc[country_filter, 'Nitrogen oxide (NOx)'].std()

df['SO2_median'] = df.loc[country_filter, 'Sulphur dioxide (SO₂) emissions'].median()
df['SO2_mean'] = df.loc[country_filter, 'Sulphur dioxide (SO₂) emissions'].mean()
df['SO2_std'] = df.loc[country_filter, 'Sulphur dioxide (SO₂) emissions'].std()

df['CO_median'] = df.loc[country_filter, 'Carbon monoxide (CO) emissions'].median()
df['CO_mean'] = df.loc[country_filter, 'Carbon monoxide (CO) emissions'].mean()
df['CO_std'] = df.loc[country_filter, 'Carbon monoxide (CO) emissions'].std()

df['OC_median'] = df.loc[country_filter, 'Organic carbon (OC) emissions'].median()
df['OC_mean'] = df.loc[country_filter, 'Organic carbon (OC) emissions'].mean()
df['OC_std'] = df.loc[country_filter, 'Organic carbon (OC) emissions'].std()

df['NMVOC_median'] = df.loc[country_filter, 'Non-methane volatile organic compounds (NMVOC) emissions'].median()
df['NMVOC_mean'] = df.loc[country_filter, 'Non-methane volatile organic compounds (NMVOC) emissions'].mean()
df['NMVOC_std'] = df.loc[country_filter, 'Non-methane volatile organic compounds (NMVOC) emissions'].std()

df['BC_median'] = df.loc[country_filter, 'Black carbon (BC) emissions'].median()
df['BC_mean'] = df.loc[country_filter, 'Black carbon (BC) emissions'].mean()
df['BC_std'] = df.loc[country_filter, 'Black carbon (BC) emissions'].std()

df['NH3_median'] = df.loc[country_filter, 'Ammonia (NH₃) emissions'].median()
df['NH3_mean'] = df.loc[country_filter, 'Ammonia (NH₃) emissions'].mean()
df['NH3_std'] = df.loc[country_filter, 'Ammonia (NH₃) emissions'].std()

# Delete the repeated entries
df = df.drop_duplicates()

# Change the null values into 0
df = df.fillna(0)

# Print the resulting DataFrame
print(df)