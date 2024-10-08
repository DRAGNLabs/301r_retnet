import json
import csv
from tqdm import tqdm

# Function to load JSON file as a dictionary


# Function to process the CSV file
def process_csv(csv_file_path, json_dict, output_csv_file_path):
    with open(csv_file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        rows = list(reader)

    with open(output_csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        counter = 0
        for row in tqdm(rows):
            if counter % 10 == 0:
                print(counter)
            if row:  # Check if row is not empty
                last_word = row[2].split()[-1]  # Get the last word of the last column
                token_ids = [v for k, v in json_dict.items() if last_word.startswith(k)]
                row.append(token_ids)  # Add matched keys to the row
            writer.writerow(row)
            counter += 1

# Example usage
json_file_path = '/home/dsg2060/fsl_groups/grp_retnet/compute/tokenizer/wikitext_large/tokenizer.json'
csv_file_path = '/home/dsg2060/301R/repos/301r_retnet/slurm/user_slurm/eval_2num_3digit_adds_5_shot.csv'
output_csv_file_path = 'tokened_2_eval_2num_3digit_adds_5_shot.csv'

token_dict = load_json_as_dict(json_file_path)
print('\nTokens aquired; Moving on to token search...\n')
process_csv(csv_file_path, token_dict, output_csv_file_path)
