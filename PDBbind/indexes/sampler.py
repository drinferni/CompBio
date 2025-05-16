import pandas as pd

# Step 1: Read the two CSV files
df1 = pd.read_csv('rs_index.csv')
df2 = pd.read_csv('casf2016_index.csv')

# Step 2: Combine the DataFrames row-wise (since they have same structure)
combined_df = pd.concat([df1, df2], ignore_index=True)

# Step 3: Shuffle the combined DataFrame
shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 4: Split into two parts
df_first_500 = shuffled_df.iloc[:500]
df_remaining = shuffled_df.iloc[500:]

# Step 5: Save to files
df_first_500.to_csv('file1.csv', index=False)
df_remaining.to_csv('file2.csv', index=False)
