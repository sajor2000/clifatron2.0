#!/usr/bin/env python3
"""
Extract numeric ranges from token_config.yaml to CSV format.
"""

import yaml
import csv
import os

# Load token config
script_dir = os.path.dirname(os.path.abspath(__file__))
token_config_path = os.path.join(script_dir, 'config', 'token_config.yaml')

with open(token_config_path, 'r') as f:
    config = yaml.safe_load(f)

# Prepare CSV data
csv_rows = []

# Header
csv_rows.append(['category', 'measurement', 'segment', 'min_value', 'max_value', 'token'])

# Process Labs
if 'labs' in config.get('tables', {}):
    for lab_name, lab_config in config['tables']['labs'].items():
        if isinstance(lab_config, dict) and 'tokenization' in lab_config:
            tokenization = lab_config['tokenization']
            if tokenization.get('enabled') and 'bins' in tokenization:
                for bin_config in tokenization['bins']:
                    csv_rows.append([
                        'labs',
                        lab_name,
                        bin_config.get('segment', ''),
                        bin_config.get('min', ''),
                        bin_config.get('max', ''),
                        bin_config.get('token', '')
                    ])

# Process Vitals
if 'vitals' in config.get('tables', {}):
    for vital_name, vital_config in config['tables']['vitals'].items():
        if isinstance(vital_config, dict) and 'tokenization' in vital_config:
            tokenization = vital_config['tokenization']
            if tokenization.get('enabled') and 'bins' in tokenization:
                for bin_config in tokenization['bins']:
                    csv_rows.append([
                        'vitals',
                        vital_name,
                        bin_config.get('segment', ''),
                        bin_config.get('min', ''),
                        bin_config.get('max', ''),
                        bin_config.get('token', '')
                    ])

# Add note about respiratory support
csv_rows.append(['# NOTE: Respiratory support ranges not currently defined in token_config.yaml', '', '', '', '', ''])
csv_rows.append(['# Suggested metrics: fio2, peep, tidal_volume, minute_ventilation, pip, plateau_pressure', '', '', '', '', ''])

# Write CSV
output_path = os.path.join(script_dir, 'config', 'numeric_ranges.csv')
with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

print(f"✓ Extracted {len(csv_rows) - 3} numeric ranges to {output_path}")
print(f"  - Labs: {sum(1 for row in csv_rows if row[0] == 'labs')} ranges")
print(f"  - Vitals: {sum(1 for row in csv_rows if row[0] == 'vitals')} ranges")
