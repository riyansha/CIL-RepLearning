import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('./data/esc10/valid_data.csv')

# Specify the column name
column_name = 'target'

# Replace all occurrences of 40 with 3 in the specified column
df['target'] = df['target'].replace(40, 3)
df['target'] = df['target'].replace(20, 2)
df['target'] = df['target'].replace(10, 1)
df['target'] = df['target'].replace(0, 0)
df['target'] = df['target'].replace(1, 4)
df['target'] = df['target'].replace(11, 5)
df['target'] = df['target'].replace(21, 6)
df['target'] = df['target'].replace(41, 7)
df['target'] = df['target'].replace(12, 8)
df['target'] = df['target'].replace(38, 9)


# Save the modified DataFrame back to a new CSV file
df.to_csv('./data/esc10/valid_esc10.csv', index=False)
