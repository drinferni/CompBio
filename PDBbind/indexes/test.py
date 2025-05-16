import csv
import random

def split_csv(input_file, test_file='test.csv', train_file='train.csv', test_size=200):
    # Read and drop the last column
    with open(input_file, 'r', newline='') as file:
        reader = csv.reader(file)
        data = [row for row in reader ]

    # Shuffle and split data
    random.shuffle(data)
    test_data = data[:test_size]
    train_data = data[test_size:]

    # Write test data
    with open(test_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(test_data)

    # Write train data
    with open(train_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(train_data)

# Example usage
split_csv('combined_output.csv')
