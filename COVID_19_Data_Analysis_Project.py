# ============================================================
# COVID-19 DATA ANALYSIS PROJECT
# ============================================================
# Author      : Abhishek Verma
# Project     : COVID-19 Data Analysis using Python & Pandas
# Dataset     : COVID-19 Clean Complete Dataset
# ============================================================

# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------
import pandas as pd

# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------
# Kaggle path
file_path = "/kaggle/input/corona-virus-report/covid_19_clean_complete.csv"

covid = pd.read_csv(file_path, parse_dates=["Date"])

print("=" * 60)
print("COVID-19 DATASET")
print("=" * 60)
print(covid)

# ------------------------------------------------------------
# 3. DATA EXPLORATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(covid.head())

print("\n" + "=" * 60)
print("LAST 5 ROWS")
print("=" * 60)
print(covid.tail())

print("\n" + "=" * 60)
print("DATA INFORMATION")
print("=" * 60)
covid.info()

print("\n" + "=" * 60)
print("NUMERICAL STATISTICS")
print("=" * 60)
print(covid.describe())

print("\n" + "=" * 60)
print("CATEGORICAL STATISTICS")
print("=" * 60)
print(covid.describe(include="O"))

# ------------------------------------------------------------
# 4. RENAME COLUMN
# ------------------------------------------------------------
covid.rename(
    columns={"Country/Region": "Location"},
    inplace=True
)

print("\n" + "=" * 60)
print("AFTER RENAMING COUNTRY/REGION TO LOCATION")
print("=" * 60)
print(covid.head())

# ------------------------------------------------------------
# 5. CHECK DUPLICATES
# ------------------------------------------------------------
duplicate_count = covid.duplicated().sum()

print("\n" + "=" * 60)
print("DUPLICATE RECORDS")
print("=" * 60)
print("Number of duplicate rows:", duplicate_count)

# ------------------------------------------------------------
# 6. CHECK MISSING VALUES
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(covid.isnull().sum())

# Province/State contains many missing values.
# It is removed because this project focuses on
# country/location-level analysis.
covid.drop("Province/State", axis=1, inplace=True)

print("\n" + "=" * 60)
print("DATA AFTER REMOVING PROVINCE/STATE")
print("=" * 60)
print(covid.head())

print("\n" + "=" * 60)
print("FINAL DATA INFORMATION")
print("=" * 60)
covid.info()

# ------------------------------------------------------------
# 7. HIGHEST AND LOWEST DEATH CASES
# ------------------------------------------------------------
death_range = covid["Deaths"].agg(["min", "max"])

print("\n" + "=" * 60)
print("MINIMUM AND MAXIMUM DEATH CASES")
print("=" * 60)
print(death_range)

max_deaths = covid["Deaths"].max()
min_deaths = covid["Deaths"].min()

print("\nLocation with maximum recorded deaths:")
print(covid[covid["Deaths"] == max_deaths])

print("\nRecords with zero deaths:")
print(covid[covid["Deaths"] == min_deaths].head(20))

# ------------------------------------------------------------
# 8. HIGHEST AND LOWEST RECOVERED CASES
# ------------------------------------------------------------
recovered_range = covid["Recovered"].agg(["min", "max"])

print("\n" + "=" * 60)
print("MINIMUM AND MAXIMUM RECOVERED CASES")
print("=" * 60)
print(recovered_range)

max_recovered = covid["Recovered"].max()

print("\nLocation with maximum recorded recovered cases:")
print(covid[covid["Recovered"] == max_recovered])

print("\nLocations with zero recovered cases:")
print(covid[covid["Recovered"] == 0]["Location"].unique())

# ------------------------------------------------------------
# 9. DEATHS BY LOCATION
# ------------------------------------------------------------
deaths_by_location = covid.groupby("Location")["Deaths"].sum()

print("\n" + "=" * 60)
print("SUM OF DEATH VALUES BY LOCATION")
print("=" * 60)
print(deaths_by_location)

# ------------------------------------------------------------
# 10. RECOVERED CASES BY LOCATION
# ------------------------------------------------------------
recovered_by_location = covid.groupby("Location")["Recovered"].sum()

print("\n" + "=" * 60)
print("SUM OF RECOVERED VALUES BY LOCATION")
print("=" * 60)
print(recovered_by_location)

# ------------------------------------------------------------
# 11. DEATHS BY WHO REGION
# ------------------------------------------------------------
deaths_by_region = covid.groupby("WHO Region")["Deaths"].sum()

print("\n" + "=" * 60)
print("SUM OF DEATHS BY WHO REGION")
print("=" * 60)
print(deaths_by_region)

# ------------------------------------------------------------
# 12. RECOVERED CASES BY WHO REGION
# ------------------------------------------------------------
recovered_by_region = covid.groupby("WHO Region")["Recovered"].sum()

print("\n" + "=" * 60)
print("SUM OF RECOVERED CASES BY WHO REGION")
print("=" * 60)
print(recovered_by_region)

# ------------------------------------------------------------
# 13. BETTER COUNTRY-LEVEL ANALYSIS
# ------------------------------------------------------------
# Confirmed, Deaths, Recovered and Active are cumulative
# values by date. Therefore, summing all dates does not
# represent the final country total.
#
# For a meaningful final-date comparison, use the latest date.

latest_date = covid["Date"].max()

latest_data = covid[covid["Date"] == latest_date]

country_totals = latest_data.groupby("Location")[
    ["Confirmed", "Deaths", "Recovered", "Active"]
].sum()

print("\n" + "=" * 60)
print("LATEST AVAILABLE DATE")
print("=" * 60)
print(latest_date)

print("\n" + "=" * 60)
print("COUNTRY TOTALS ON LATEST DATE")
print("=" * 60)
print(country_totals)

# ------------------------------------------------------------
# 14. TOP 10 LOCATIONS BY CONFIRMED CASES
# ------------------------------------------------------------
top_10_confirmed = country_totals.sort_values(
    "Confirmed",
    ascending=False
).head(10)

print("\n" + "=" * 60)
print("TOP 10 LOCATIONS BY CONFIRMED CASES")
print("=" * 60)
print(top_10_confirmed)

# ------------------------------------------------------------
# 15. TOP 10 LOCATIONS BY DEATHS
# ------------------------------------------------------------
top_10_deaths = country_totals.sort_values(
    "Deaths",
    ascending=False
).head(10)

print("\n" + "=" * 60)
print("TOP 10 LOCATIONS BY DEATHS")
print("=" * 60)
print(top_10_deaths)

# ------------------------------------------------------------
# 16. TOP 10 LOCATIONS BY RECOVERED CASES
# ------------------------------------------------------------
top_10_recovered = country_totals.sort_values(
    "Recovered",
    ascending=False
).head(10)

print("\n" + "=" * 60)
print("TOP 10 LOCATIONS BY RECOVERED CASES")
print("=" * 60)
print(top_10_recovered)

# ------------------------------------------------------------
# 17. SAVE ANALYSIS OUTPUTS
# ------------------------------------------------------------
country_totals.to_csv("country_totals_latest_date.csv")
deaths_by_region.to_csv("deaths_by_region.csv")
recovered_by_region.to_csv("recovered_by_region.csv")

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)
print("Generated files:")
print("1. country_totals_latest_date.csv")
print("2. deaths_by_region.csv")
print("3. recovered_by_region.csv")
