# COVID-19 Data Analysis using Pandas
# Author: Abhishek Verma

import pandas as pd

# ---------------------------------------------------------
# 1. Import dataset
# ---------------------------------------------------------
# For Kaggle:
FILE_PATH = "/kaggle/input/corona-virus-report/covid_19_clean_complete.csv"

covid = pd.read_csv(FILE_PATH, parse_dates=["Date"])

print("\n===== DATASET =====")
print(covid)

# ---------------------------------------------------------
# 2. Explore data
# ---------------------------------------------------------
print("\n===== FIRST 5 ROWS =====")
print(covid.head())

print("\n===== LAST 5 ROWS =====")
print(covid.tail())

print("\n===== DATA INFORMATION =====")
print(covid.info())

print("\n===== STATISTICAL SUMMARY =====")
print(covid.describe())

print("\n===== CATEGORICAL SUMMARY =====")
print(covid.describe(include="O"))

# ---------------------------------------------------------
# 3. Rename column
# ---------------------------------------------------------
covid.rename(columns={"Country/Region": "Location"}, inplace=True)

print("\n===== AFTER RENAMING COUNTRY/REGION =====")
print(covid.head())

# ---------------------------------------------------------
# 4. Check duplicates and missing values
# ---------------------------------------------------------
print("\n===== DUPLICATE RECORDS =====")
print("Number of duplicate rows:", covid.duplicated().sum())

print("\n===== MISSING VALUES =====")
print(covid.isnull().sum())

# Province/State has many missing values.
# It is removed because country-level analysis is our focus.
covid.drop("Province/State", axis=1, inplace=True)

print("\n===== DATA AFTER DROPPING PROVINCE/STATE =====")
print(covid.head())

print("\n===== FINAL DATA INFORMATION =====")
print(covid.info())

# ---------------------------------------------------------
# 5. Highest and lowest death cases
# ---------------------------------------------------------
print("\n===== MINIMUM AND MAXIMUM DEATHS =====")
print(covid["Deaths"].agg(["min", "max"]))

max_deaths = covid["Deaths"].max()
min_deaths = covid["Deaths"].min()

print("\n===== COUNTRY/LOCATION WITH MAXIMUM DEATHS =====")
print(covid[covid["Deaths"] == max_deaths])

print("\n===== RECORDS WITH ZERO DEATHS =====")
print(covid[covid["Deaths"] == min_deaths].head(20))

# ---------------------------------------------------------
# 6. Highest and lowest recovered cases
# ---------------------------------------------------------
print("\n===== MINIMUM AND MAXIMUM RECOVERED =====")
print(covid["Recovered"].agg(["min", "max"]))

max_recovered = covid["Recovered"].max()

print("\n===== LOCATION WITH MAXIMUM RECOVERED CASES =====")
print(covid[covid["Recovered"] == max_recovered])

print("\n===== LOCATIONS WITH ZERO RECOVERED CASES =====")
print(covid[covid["Recovered"] == 0]["Location"].unique())

# ---------------------------------------------------------
# 7. Deaths by location
# ---------------------------------------------------------
print("\n===== TOTAL RECORDED DEATH VALUES BY LOCATION =====")
deaths_by_location = covid.groupby("Location")["Deaths"].sum()
print(deaths_by_location)

# ---------------------------------------------------------
# 8. Recovered cases by location
# ---------------------------------------------------------
print("\n===== TOTAL RECORDED RECOVERED VALUES BY LOCATION =====")
recovered_by_location = covid.groupby("Location")["Recovered"].sum()
print(recovered_by_location)

# ---------------------------------------------------------
# 9. Deaths by WHO region
# ---------------------------------------------------------
print("\n===== DEATHS BY WHO REGION =====")
deaths_by_region = covid.groupby("WHO Region")["Deaths"].sum()
print(deaths_by_region)

# ---------------------------------------------------------
# 10. Recovered cases by WHO region
# ---------------------------------------------------------
print("\n===== RECOVERED CASES BY WHO REGION =====")
recovered_by_region = covid.groupby("WHO Region")["Recovered"].sum()
print(recovered_by_region)

# ---------------------------------------------------------
# 11. Recommended analysis:
# Latest-date country totals
# ---------------------------------------------------------
# Deaths and recovered cases are cumulative by date.
# Therefore, summing all dates can double-count cumulative values.
latest_date = covid["Date"].max()

latest = covid[covid["Date"] == latest_date].copy()

country_totals = latest.groupby("Location")[
    ["Confirmed", "Deaths", "Recovered", "Active"]
].sum()

print("\n===== LATEST DATE =====")
print(latest_date)

print("\n===== COUNTRY TOTALS ON LATEST DATE =====")
print(country_totals)

# ---------------------------------------------------------
# 12. Top 10 locations by confirmed cases
# ---------------------------------------------------------
top_10_confirmed = country_totals.sort_values(
    "Confirmed", ascending=False
).head(10)

print("\n===== TOP 10 LOCATIONS BY CONFIRMED CASES =====")
print(top_10_confirmed)

# ---------------------------------------------------------
# 13. Top 10 locations by deaths
# ---------------------------------------------------------
top_10_deaths = country_totals.sort_values(
    "Deaths", ascending=False
).head(10)

print("\n===== TOP 10 LOCATIONS BY DEATHS =====")
print(top_10_deaths)

# ---------------------------------------------------------
# 14. Top 10 locations by recovered cases
# ---------------------------------------------------------
top_10_recovered = country_totals.sort_values(
    "Recovered", ascending=False
).head(10)

print("\n===== TOP 10 LOCATIONS BY RECOVERED CASES =====")
print(top_10_recovered)

# ---------------------------------------------------------
# 15. Save useful outputs
# ---------------------------------------------------------
country_totals.to_csv("country_totals_latest_date.csv")

deaths_by_region.to_csv("deaths_by_region.csv")
recovered_by_region.to_csv("recovered_by_region.csv")

print("\n===== ANALYSIS COMPLETED =====")
print("Output files created:")
print("- country_totals_latest_date.csv")
print("- deaths_by_region.csv")
print("- recovered_by_region.csv")
