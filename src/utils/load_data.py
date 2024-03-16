
import pandas as pd
file_path = "data/recipes/full_dataset.csv"

# Load the .csv file into a DataFrame
# Specify the delimiter if it's not a comma
# Specify the encoding if it's not 'utf-8'
# Use the 'usecols' parameter to select specific columns
# Use the 'dtype' parameter to specify data types for columns
df = pd.read_csv(file_path, delimiter=',', encoding='utf-8')

# Create a smaller test list from the DataFrame
test_list = df.sample(n=100)

# Save the test list to a new CSV file
test_list.to_csv('data/recipes/test_data.csv', index=False)


# Print the first 5 rows of the DataFrame
# print(df.head())
