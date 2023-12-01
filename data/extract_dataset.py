import json

file_path = 'WLASL_v0.3.json'

with open(file_path) as file:
    data = json.load(file)

# JSON with the top 100 glosses
top_100_entries = sorted(data, key=lambda entry: len(entry['instances']), reverse=True)[:100]

# Write to a new JSON file
output_file_path = 'ASL_data.json'
with open(output_file_path, 'w') as output_file:
    json.dump(top_100_entries, output_file, indent=2)
