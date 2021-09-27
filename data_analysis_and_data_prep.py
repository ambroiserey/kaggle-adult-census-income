import pandas as pd
# Display all the columns
pd.set_option("display.max_columns", None)
import numpy as np
import matplotlib.pyplot as plt

# Read the census_income_learn.csv and replace "?" by "Nan"
df_learn = pd.read_csv(r"census_income_learn.csv", sep=',\s', na_values=["?"])

# Read the census_income_test.csv and replace "?" by "Nan"
df_test = pd.read_csv(r"census_income_test.csv", sep=',\s', na_values=["?"])

# Function to convert the age feature from continuous to categorical
def age_category(age):
    if age < 16:
        category = "0-15"
    elif 20 > age > 15:
        category = "16-19"
    elif 25 > age > 19:
        category = "20-24"
    elif 35 > age > 24:
        category = "25-34"
    elif 45 > age > 34:
        category = "35-44"
    elif 55 > age > 44:
        category = "45-54"
    elif 65 > age > 54:
        category = "55-64"
    elif age > 64:
        category = "65+"
    return category

# I defined my categories following the questions asked in the census
    
# Function to convert the capital gains feature from continuous to categorical
def capital_gains_category(capital_gains):
    if capital_gains == 0:
        category = "no gain"
    elif  10001 > capital_gains > 0:
        category = "between USD 1 and USD 10 000"
    elif 20001 > capital_gains > 10000:
        category = "between USD 10 000 and USD 20 000"
    elif capital_gains > 20000:
        category = "over USD 20 000"
    return category

# Function to convert the capital losses feature from continuous to categorical
def capital_losses_category(capital_losses):
    if capital_losses == 0:
        category = "no losses"
    elif 501 > capital_losses > 0:
        category = "between USD 1 and USD 500"
    elif 1001 > capital_losses > 500:
        category = "between USD 500 and USD 1 000"
    elif capital_losses > 1000:
        category = "over USD 1 000"
    return category
        
# Function to convert the dividends from stocks feature from continuous to categorical
def dividends_category(dividends):
    if dividends == 0:
        category = "no dividends"
    elif 1001 > dividends > 0:
        category = "between USD 1 and USD 1 000"
    elif 5001 > dividends > 1000:
        category = "between USD 1 000 and USD 5 000"
    elif dividends > 5000:
        category = "over USD 5 000"
    return category
    
def cleaning(df):

    # Add the name of the columns to the pandas dataframe
    df.columns = ["age", 
                  "class of worker", 
                  "detailed industry recode", 
                  "detailed occupation recode", 
                  "education", 
                  "wage per hour", 
                  "enrolled in edu inst last wk", 
                  "marital status", 
                  "major industry code", 
                  "major occupation code",
                  "race",
                  "hispanic origin",
                  "sex",
                  "member of a labor union",
                  "reason for unemployment",
                  "full or part time employment stat",
                  "capital gains",
                  "capital losses",
                  "dividends from stocks",
                  "tax filer stat",
                  "region of previous residence",
                  "state of previous residence",
                  "detailed household and family stat",
                  "detailed household summary in household",
                    
                  # the missing column, to be dropped later
                  "instance weight",
                    
                  "migration code-change in msa",
                  "migration code-change in reg",
                  "migration code-move within reg",
                  "live in this house 1 year ago",
                  "migration prev res in sunbelt",
                  "num persons worked for employer",
                  "family members under 18",
                  "country of birth father",
                  "country of birth mother",
                  "country of birth self",
                  "citizenship",
                  "own business or self employed",
                  "fill inc questionnaire for veteran's admin",
                  "veterans benefits",
                  "weeks worked in year",
                  "year",
                  "class"
                       ]

    # Drop all the duplicates
    df = df.drop_duplicates()

    # Create a dictionary with the number of "NaN" per column
    dict_nans = dict(df.isnull().sum())

    # Initialize the columns_to_drop list
    columns_to_drop = []

    # Add to the above mentioned list all the columns that have more than 40% "Nan"
    for key, value in dict_nans.items():
        if value / df.shape[0] > 0.4:
            columns_to_drop.append(key)

    # Drop all the columns that have more than 40% "Nan"
    df = df.drop(columns = columns_to_drop)
    
    # Drop all the rows containing at least one "NaN"
    df = df.dropna()
    
    return df

def preparation(df):

    # Prepare the age feature for feature encoding
    df["age"] = df["age"].apply(lambda x: age_category(x))
    
    # Prepare the capital gains feature for feature encoding
    df["capital gains"] = df["capital gains"].apply(lambda x: capital_gains_category(x))

    # Prepare the capital losses feature for feature encoding
    df["capital losses"] = df["capital losses"].apply(lambda x: capital_losses_category(x))
    
    # Prepare the dividends from stocks feature for feature encoding
    df["dividends from stocks"] = df["dividends from stocks"].apply(lambda x: dividends_category(x))

    # Drop the wage per hour and weeks worked in year columns, see analysis later
    df = df.drop(columns = ["wage per hour", "weeks worked in year"])

    # Drop all the columns that are not relevant for predicting the class or that are redundant
    df = df.drop(columns = ["num persons worked for employer", "instance weight", "fill inc questionnaire for veteran's admin"])
    
    return df
    
# Clean the data
df_learn = cleaning(df_learn)

# Plot the number of instances/values per class to check sample imbalance
ax = df_learn["class"].value_counts().plot(kind = "bar", figsize = (10, 6), fontsize = 13, color = "#087E8B")
ax.set_title("Number of instances/values per class", size=20, pad=30)

# Compare the mean and the std of capital gains and capital losses to create a scale for capital losses
print("Mean capital gains", round(df_learn["capital gains"].mean(), 0))
print("Mean capital losses", round(df_learn["capital losses"].mean(), 0))
print("Std capital gains", round(df_learn["capital gains"].std(), 0))   
print("Std capital losses", round(df_learn["capital losses"].std(), 0)) 

# The num persons worked for employer feature is already encoded
print("Size of the firm", df_learn["num persons worked for employer"].unique())

# The fill inc questionnaire for veteran's admin feature is redundant with the veterans benefits feature
print("Value counts for fill inc questionnaire for veteran's admin", df_learn["fill inc questionnaire for veteran's admin"].value_counts())

# Determine the weighted mean wage per hour, it is still way above the real one (around 22 dollars at the time)
weighted_mean_wage_per_hour = (df_learn["wage per hour"]*df_learn["instance weight"]).sum()/df_learn["instance weight"].sum()

# Print the weighted mean wage per hour
print("Weighted mean wage per hour", weighted_mean_wage_per_hour)

# Create a temporary df_wage_per_hour with the columns wage per hour and instance weight of df_learn
df_wage_per_hour = df_learn[["wage per hour", "instance weight"]]

# Group by wage per hour and sum
df_wage_per_hour = df_wage_per_hour.groupby(["wage per hour"]).sum()

# Reset the index of df_wage_per_hour
df_wage_per_hour = df_wage_per_hour.rename_axis("wage per hour").reset_index()

# Transform the instance weight proportionaly to the whole population
df_wage_per_hour["instance weight"] = df_wage_per_hour["instance weight"].apply(lambda x: float(x))
df_wage_per_hour["instance weight"] = df_wage_per_hour["instance weight"].apply(lambda x: int(x)/df_wage_per_hour["instance weight"].sum())

# Sort the value by descending order
df_wage_per_hour = df_wage_per_hour.sort_values(by = "wage per hour", ascending=False)

# Build a scatterplot to better understand the distribution of the wages per hour
ax = df_wage_per_hour.plot(x = "wage per hour", y = "instance weight", kind = "scatter")
ax.ticklabel_format(useOffset = False, style = "plain")
plt.xlabel("wage per hour")
plt.ylabel("proportion of Americans")
plt.show

# Create a temporary df_number_of_weeks with the columns weeks worked in year and instance weight of df_learn
df_number_of_weeks = df_learn[["weeks worked in year", "instance weight"]]

# Group by weeks worked in year and sum
df_number_of_weeks = df_number_of_weeks.groupby(["weeks worked in year"]).sum()

# Reset the index of df_number_of_weeks
df_number_of_weeks = df_number_of_weeks.rename_axis("weeks worked in year").reset_index()

# Transform the instance weight proportionaly to the whole population
df_number_of_weeks["instance weight"] = df_number_of_weeks["instance weight"].apply(lambda x: float(x))
df_number_of_weeks["instance weight"] = df_number_of_weeks["instance weight"].apply(lambda x: int(x)/df_number_of_weeks["instance weight"].sum())

# Sort the value by descending order
df_number_of_weeks = df_number_of_weeks.sort_values(by = "weeks worked in year", ascending=False)

# Build a scatterplot to better understand the distribution of weeks worked in year
ax = df_number_of_weeks.plot(x = "weeks worked in year", y = "instance weight", kind = "scatter")
ax.ticklabel_format(useOffset = False, style = "plain")
plt.xlabel("weeks worked in year")
plt.ylabel("proportion of Americans")
plt.show

# Prepare the data
df_learn = preparation(df_learn)

# Clean the data using the same function as for df_learn
df_test = cleaning(df_test)

# Prepare the data using the same function as for df_learn
df_test = preparation(df_test)
