import pandas as pd
import random

# Replace 'esc50.csv' with the path to your ESC-10 CSV file
data = pd.read_csv('/home/riyansha/meta-sc/data/esc10/esc10.csv')


# Set the random seed for reproducibility
random.seed(42)

# Define the split ratios
train_ratio = 0.7
test_ratio = 0.15
valid_ratio = 0.15

# Calculate the number of samples for each split
total_samples = len(data)
print("length",total_samples)
num_train = int(train_ratio * total_samples)
num_test = int(test_ratio * total_samples)

# Shuffle the data to randomize the split
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the data
train_data = data[:num_train]
test_data = data[num_train:num_train + num_test]
valid_data = data[num_train + num_test:]

train_data.to_csv('./data/esc10/train_data.csv', index=False)
test_data.to_csv('./data/esc10/test_data.csv', index=False)
valid_data.to_csv('./data/esc10/valid_data.csv', index=False)