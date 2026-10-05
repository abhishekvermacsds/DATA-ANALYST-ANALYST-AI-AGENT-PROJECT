# DATA-ANALYST-ANALYST-AI-AGENT-PROJECT
AI AGENTS DATA SCIENCE
# COVID-19 Data Analysis Using Python & Pandas

## 📌 Project Overview

This project performs an exploratory data analysis (EDA) of COVID-19 cases using Python and Pandas.

The dataset contains COVID-19 information for different countries and regions, including:

- Confirmed cases
- Deaths
- Recovered cases
- Active cases
- Geographic coordinates
- WHO regions
- Date-wise COVID-19 records

The main objective is to clean, explore, summarize, and analyze the COVID-19 dataset to identify important patterns and statistics.

---

## 🛠️ Technologies Used

- Python
- Pandas
- Jupyter Notebook / Kaggle Notebook
- CSV Dataset
- Exploratory Data Analysis (EDA)

---

## 📂 Dataset

The dataset used in this project is:

`covid_19_clean_complete.csv`

It contains **49,068 records** and initially has **10 columns**.

Important columns include:

| Column | Description |
|---|---|
| Province/State | State or province name |
| Country/Region | Country name |
| Lat | Latitude |
| Long | Longitude |
| Date | Date of observation |
| Confirmed | Confirmed COVID-19 cases |
| Deaths | Number of deaths |
| Recovered | Number of recovered cases |
| Active | Active cases |
| WHO Region | WHO geographical region |

---

## 🔍 Data Analysis Performed

### 1. Importing Pandas

```python
import pandas as pd
```

### 2. Reading the Dataset

```python
covid = pd.read_csv(
    '/kaggle/input/corona-virus-report/covid_19_clean_complete.csv',
    parse_dates=['Date']
)
```

The `parse_dates` parameter converts the `Date` column into a datetime data type.

---

## 3. Exploring the Dataset

### First Five Records

```python
covid.head()
```

### Last Five Records

```python
covid.tail()
```

### Dataset Information

```python
covid.info()
```

The original dataset contains:

- 49,068 rows
- 10 columns
- 78 unique province/state values
- 187 countries/locations
- 6 WHO regions

---

## 4. Statistical Summary

```python
covid.describe()
```

This provides statistical information such as:

- Mean
- Minimum
- Maximum
- Standard deviation
- Quartiles

for numerical columns.

Categorical information was analyzed using:

```python
covid.describe(include='O')
```

---

## 5. Renaming Columns

The `Country/Region` column was renamed to `Location` to make the dataset easier to work with.

```python
covid.rename(
    columns={'Country/Region': 'Location'},
    inplace=True
)
```

---

## 6. Handling Missing Values

Missing values were checked using:

```python
covid.isnull().sum()
```

The `Province/State` column contained many missing values because several records were reported only at the country level.

Since province/state information was not required for the main analysis, it was removed:

```python
covid.drop('Province/State', axis=1, inplace=True)
```

After removal, the dataset contains **9 columns**.

---

## 7. Checking Duplicate Records

```python
covid.duplicated().sum()
```

Result:

```text
0
```

Therefore, no duplicate records were found.

---

# 📊 Analysis

## 8. Highest and Lowest Death Cases

The minimum and maximum death values were found using:

```python
covid['Deaths'].agg(['min', 'max'])
```

Maximum deaths:

**148,011**

The highest value occurred for:

**US — 27 July 2020**

```python
covid[covid['Deaths'] == 148011]
```

This record contained:

- Confirmed: 4,290,259
- Deaths: 148,011
- Recovered: 1,325,804
- Active: 2,816,444

---

## 9. Highest Recovered Cases

The maximum number of recovered cases was:

**1,846,641**

It was recorded for:

**Brazil — 27 July 2020**

```python
covid[covid['Recovered'] == 1846641]
```

---

## 10. Deaths by Location

Deaths were aggregated by location using:

```python
covid.groupby('Location')['Deaths'].sum()
```

This calculates the total recorded deaths across all available dates for each location.

---

## 11. Recovered Cases by Location

Recovered cases were aggregated by location:

```python
covid.groupby('Location')['Recovered'].sum()
```

This provides the total recorded recovered cases for each location across the dataset.

---

# 🌍 Regional Analysis

## 12. Total Deaths by WHO Region

```python
covid.groupby('WHO Region')['Deaths'].sum()
```

| WHO Region | Total Deaths |
|---|---:|
| Africa | 439,978 |
| Americas | 19,359,292 |
| Eastern Mediterranean | 1,924,029 |
| Europe | 19,271,040 |
| South-East Asia | 1,458,134 |
| Western Pacific | 932,430 |

### Observation

The **Americas** recorded the highest cumulative sum of deaths in this dataset, followed closely by **Europe**.

---

## 13. Total Recovered Cases by WHO Region

```python
covid.groupby('WHO Region')['Recovered'].sum()
```

| WHO Region | Total Recovered |
|---|---:|
| Africa | 11,193,730 |
| Americas | 157,069,444 |
| Eastern Mediterranean | 48,050,703 |
| Europe | 123,202,075 |
| South-East Asia | 30,030,327 |
| Western Pacific | 18,861,950 |

### Observation

The **Americas** recorded the highest total recovered cases, followed by **Europe**.

---

# 💡 Key Findings

1. The dataset contains **49,068 COVID-19 records**.
2. The dataset covers the period from **22 January 2020 to 27 July 2020**.
3. Data is available for **187 locations**.
4. No duplicate records were found.
5. `Province/State` contained substantial missing data and was removed.
6. The highest recorded death count in a single row was **148,011** for the US.
7. The highest recorded recovered count in a single row was **1,846,641** for Brazil.
8. The Americas had the highest summed deaths among the WHO regions.
9. The Americas also had the highest summed recovered cases.

---

# ⚠️ Important Note About Aggregation

The `Deaths` and `Recovered` values in this dataset are cumulative values for each date.

Therefore, simply using:

```python
covid.groupby('Location')['Deaths'].sum()
```

adds cumulative daily values together. It should **not** be interpreted as the actual final number of deaths in a country.

For final country-level totals, the last available date should normally be selected:

```python
latest_date = covid['Date'].max()

latest = covid[covid['Date'] == latest_date]

country_totals = latest.groupby('Location')[
    ['Confirmed', 'Deaths', 'Recovered', 'Active']
].sum()
```

This provides a more meaningful final-date comparison.

---

# 🚀 Future Improvements

The project can be extended by adding:

- COVID-19 trend analysis
- Country-wise comparisons
- Death rate calculation
- Recovery rate calculation
- Time-series visualization
- Top 10 countries analysis
- WHO region comparison charts
- Matplotlib visualizations
- Seaborn visualizations
- Interactive dashboards using Power BI/Tableau
- Geographical COVID-19 maps

---

# 📁 Project Files

```text
COVID-19-Data-Analysis/
│
├── README.md
├── covid_19_analysis.ipynb
├── requirements.txt
├── data/
│   └── covid_19_clean_complete.csv
├── images/
└── output/
```

---

# 👨‍💻 Author

**Abhishek Verma**

Data Analysis / Data Science Project

---

## ⭐ Conclusion

This project demonstrates the use of **Pandas for data loading, cleaning, exploration, aggregation, and statistical analysis**.

It provides a basic understanding of COVID-19 patterns across countries and WHO regions and can be further developed into a complete data visualization and dashboard project.
