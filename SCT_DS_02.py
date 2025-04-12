import pandas as pd

try:
    df = pd.read_csv('train.csv')
    display(df.head())
    display(df.info())
except FileNotFoundError:
    print("Error: 'train.csv' not found. Please ensure the file is in the current directory or provide the correct path.")
except Exception as e:
    print(f"An error occurred: {e}")
# Data Shape
print("Data Shape:", df.shape)

# Data Types
print("\nData Types:")
print(df.dtypes)

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Descriptive Statistics (Numerical Features)
print("\nDescriptive Statistics (Numerical Features):")
print(df.describe())

# Value Counts (Categorical Features)
categorical_cols = df.select_dtypes(include=['object']).columns
print("\nValue Counts (Categorical Features):")
for col in categorical_cols:
    print(f"\nValue counts for {col}:")
    print(df[col].value_counts())

# Distribution Overview
print("\nDistribution Overview:")
print("Numerical Features:")
for col in df.select_dtypes(include=['number']).columns:
    print(f"- {col}:  Distribution needs further investigation (histogram, boxplot).")
print("\nCategorical Features:")
for col in df.select_dtypes(include=['object']).columns:
    print(f"- {col}: Distribution needs further investigation (bar chart).")
print("\nPotential Issues:")
print("- Missing values in 'Age', 'Cabin', and 'Embarked' columns need to be addressed.")
print("- The 'Cabin' column has a high number of missing values, consider dropping or imputation.")
print("- Investigate the distribution of 'Fare' for potential outliers.")
# Impute missing 'Age' with the median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Impute missing 'Embarked' with the most frequent value
most_frequent_embarked = df['Embarked'].mode()[0]
df['Embarked'].fillna(most_frequent_embarked, inplace=True)

# Create 'Cabin_assigned' and drop 'Cabin'
df['Cabin_assigned'] = df['Cabin'].notna().astype(int)
df.drop('Cabin', axis=1, inplace=True)

# Handle outliers in 'Fare' using winsorization (capping)
from scipy.stats.mstats import winsorize

df['Fare'] = winsorize(df['Fare'], limits=[0.0, 0.05]) # Cap the top 5%

# Ensure consistency in 'Sex' column
df['Sex'] = df['Sex'].str.lower()

display(df.head())
display(df.info())
import pandas as pd

# Convert 'Age' to integer
try:
    df['Age'] = df['Age'].astype(int)
except ValueError as e:
    print(f"Error converting 'Age' to integer: {e}")
    # Handle the error, e.g., by removing rows with invalid 'Age' values or imputing them
    # For this example, we'll remove the rows with problematic values
    # df = df[pd.to_numeric(df['Age'], errors='coerce').notnull()]
    # df['Age'] = df['Age'].astype(int)


# One-hot encode 'Sex' and 'Embarked'
df_encoded = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Drop unnecessary columns
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Sex_male', 'Embarked_Q']
try:
    df_final = df_encoded.drop(columns=columns_to_drop)
except KeyError as e:
    print(f"Error dropping columns: {e}")
    # Handle the key error, e.g., by checking the existence of columns

# Display the first few rows and info
display(df_final.head())
display(df_final.info())
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate and visualize the correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = df_final.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Group-by operations (example with 'Pclass')
print("\nSurvival Rate by Passenger Class:")
display(df_final.groupby('Pclass')['Survived'].mean())

print("\nDescriptive Statistics by Passenger Class:")
display(df_final.groupby('Pclass').agg({'Age': ['mean', 'median', 'std'],
                                        'Fare': ['mean', 'median', 'std']}))

# Relationship between 'Age' and 'Survived'
df_final['Age_band'] = pd.cut(df_final['Age'], bins=[0, 18, 30, 50, 80],
                             labels=['Child', 'Young Adult', 'Adult', 'Senior'])
print("\nSurvival Rate by Age Band:")
display(df_final.groupby('Age_band')['Survived'].mean())

# Relationship between 'Fare' and 'Survived'
df_final['Fare_band'] = pd.qcut(df_final['Fare'], 4, labels=False)
print("\nSurvival Rate by Fare Band:")
display(df_final.groupby('Fare_band')['Survived'].mean())

plt.figure(figsize=(10, 6))
sns.boxplot(x='Fare_band', y='Fare', hue='Survived', data=df_final)
plt.title('Fare Distribution by Survival Status and Fare Band')
plt.show()

print("\nSurvival Rate by Fare and Pclass:")
fare_pclass_survival = df_final.groupby(['Fare_band', 'Pclass'])['Survived'].mean().unstack()
display(fare_pclass_survival)
