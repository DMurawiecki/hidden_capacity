import json

# Read the notebook
with open('notebooks/text_ppl_vs_compression.ipynb', 'r') as f:
    notebook = json.load(f)

# Fix the numpy integer issue in the color map access
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            if 'global_color_map[max_length]' in line:
                # Replace with int(max_length) to convert numpy int to Python int
                cell['source'][i] = line.replace('global_color_map[max_length]', 'global_color_map[int(max_length)]')

# Write the fixed notebook
with open('notebooks/text_ppl_vs_compression.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Fixed the numpy integer issue in the notebook!") 