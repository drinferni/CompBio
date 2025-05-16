def add_comma_if_two_commas(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    modified_lines = []
    for line in lines:
        line = line.rstrip('\n')  # Remove newline for accurate counting
        if line.count(',') == 1:
            line += ',None'
        modified_lines.append(line + '\n')  # Add newline back

    with open(filepath, 'w') as file:
        file.writelines(modified_lines)

# Example usage
add_comma_if_two_commas('combined_output.csv')
