"""
Unemployment Rate Analysis
Author: [Your Name]
Date: 2026-03-01
Description:
This script analyzes global unemployment rates, visualizes trends, 
and highlights the impact of COVID-19 on unemployment.
"""

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load dataset
data = pd.read_csv('unemployment.csv')  # Replace with your dataset path
print("First 5 rows of dataset:")
print(data.head())

# Step 2: Explore dataset
print("\nDataset info:")
print(data.info())

print("\nMissing values per column:")
print(data.isnull().sum())

print("\nSummary statistics:")
print(data.describe())

# Step 3: Visualize unemployment trends for a specific country
country = 'USA'
country_data = data[data['Country'] == country]
plt.figure(figsize=(10,5))
plt.plot(country_data['Year'], country_data['Unemployment_rate'], marker='o', color='blue')
plt.title(f'Unemployment Rate Over Time - {country}')
plt.xlabel('Year')
plt.ylabel('Unemployment Rate (%)')
plt.grid(True)
plt.show()

# Step 4: Compare unemployment trends across multiple countries
top_countries = ['USA', 'India', 'Germany', 'Brazil']
plt.figure(figsize=(12,6))
for c in top_countries:
    temp = data[data['Country'] == c]
    plt.plot(temp['Year'], temp['Unemployment_rate'], marker='o', label=c)
plt.title('Unemployment Rate Comparison Across Countries')
plt.xlabel('Year')
plt.ylabel('Unemployment Rate (%)')
plt.legend()
plt.grid(True)
plt.show()

# Step 5: Distribution of unemployment rates
plt.figure(figsize=(8,5))
sns.histplot(data['Unemployment_rate'], bins=20, kde=True, color='green')
plt.title('Distribution of Unemployment Rates')
plt.xlabel('Unemployment Rate (%)')
plt.show()

# Step 6: Analyze COVID-19 impact (2019-2021)
covid_years = [2019, 2020, 2021]
covid_data = data[data['Year'].isin(covid_years)]
avg_unemployment = covid_data.groupby('Year')['Unemployment_rate'].mean()
print("\nAverage Unemployment Rate During COVID-19:")
print(avg_unemployment)

plt.figure(figsize=(8,5))
avg_unemployment.plot(kind='bar', color='salmon')
plt.title('Average Unemployment Rate During COVID-19')
plt.ylabel('Average Unemployment Rate (%)')
plt.grid(axis='y')
plt.show()
