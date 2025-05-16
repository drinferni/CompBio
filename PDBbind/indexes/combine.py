import pandas as pd

def process_csv(file1, file2, output_file='combined_output.csv'):
    # Read the first CSV file and drop the last column
    df1 = pd.read_csv(file1)
    df1 = df1.iloc[:, :-1]

    # Read the second CSV file
    df2 = pd.read_csv(file2)

    # Filter out rows in df2 that have the same value in the first column as df1
    first_column_name = df1.columns[0]
    df2_filtered = df2[~df2[first_column_name].isin(df1[first_column_name])]

    # Combine the DataFrames
    combined_df = pd.concat([df1, df2_filtered], ignore_index=True)

    # Save the combined DataFrame to a new CSV
    combined_df.to_csv(output_file, index=False)
    print(f"Filtered and combined CSV saved to {output_file}")

# Example usage:
process_csv('rs_index.csv', 'casf2016_index.csv')
