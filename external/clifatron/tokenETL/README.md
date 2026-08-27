# tokenETL - CLIF Data Cohort Loading Pipeline

A standalone script for loading CLIF (Common Longitudinal ICU data Format) tables using `clifpy` table objects. This is the first step in the tokenization pipeline for foundation model training.

## Features

- **Direct clifpy integration**: Uses `clifpy.tables` objects (Patient, Hospitalization, Adt, HospitalDiagnosis)
- **Configuration-driven**: Define tables and columns in `token_config.yaml`
- **Logging**: All steps logged to `{output_dir}/tokenETL.log`
- **Pandas DataFrames**: Keeps data as pandas (no Polars conversion)
- **Intermediate tables**: Saves loaded tables to `{output_dir}/intermediate_tables/`
- **CLI interface**: Simple command-line execution with `uv run`

## Prerequisites

- Python 3.8+
- `clifpy` installed via `uv`
- Valid `clif_config.json` with CLIF data configuration

## Configuration Files

### 1. `clif_config.json` (Required)

Main configuration file with data location and settings:

```json
{
  "site": "xyz",
  "data_directory": "/path/to/clif/data",
  "filetype": "parquet",
  "timezone": "US/Central",
  "output_dir": "/path/to/output"
}
```

**Fields:**
- `site`: Site identifier
- `data_directory`: Path to CLIF data directory
- `filetype`: File format (parquet, csv, etc.)
- `timezone`: Timezone for datetime columns
- `output_dir`: Directory for logs and intermediate tables

### 2. `config/token_config.yaml` (Included)

Defines which tables to load and their columns. Located in `tokenETL/config/` directory:

```yaml
tables:
  patient:
    columns:
      - patient_id
      - sex_category
      - race_category
      - ethnicity_category

  hospitalization:
    columns:
      - hospitalization_id
      - patient_id
      - admission_dttm
      - discharge_dttm
      - age_at_admission
      - discharge_category

  adt:
    columns:
      - hospitalization_id
      - location_category
      - in_dttm
      - out_dttm

  hospital_diagnosis:
    columns:
      - hospitalization_id
      - diagnosis_code
      - diagnosis_code_format
      - diagnosis_primary
```

## Usage

### Basic Usage

```bash
uv run tokenETL/main.py --config clif_config.json
```

This will:
1. Load configuration from `clif_config.json`
2. Set up logging to `{output_dir}/tokenETL.log`
3. Load tables defined in `config/token_config.yaml`
4. Print table shapes
5. Save intermediate tables to `{output_dir}/intermediate_tables/`

### Example Output

```
🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀
STARTING tokenETL - COHORT LOADING
🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀

2025-10-12 10:15:30 - tokenETL - INFO - ============================================================
2025-10-12 10:15:30 - tokenETL - INFO - tokenETL Pipeline - Cohort Loading
2025-10-12 10:15:30 - tokenETL - INFO - ============================================================
2025-10-12 10:15:30 - tokenETL - INFO - Log file: /path/to/output/tokenETL.log
2025-10-12 10:15:30 - tokenETL - INFO - Site: xyz
2025-10-12 10:15:30 - tokenETL - INFO - Data directory: /path/to/clif/data
2025-10-12 10:15:30 - tokenETL - INFO - File type: parquet
2025-10-12 10:15:30 - tokenETL - INFO - Timezone: US/Central
2025-10-12 10:15:30 - tokenETL - INFO - Output directory: /path/to/output

2025-10-12 10:15:30 - tokenETL - INFO - ============================================================
2025-10-12 10:15:30 - tokenETL - INFO - LOADING TABLES
2025-10-12 10:15:30 - tokenETL - INFO - ============================================================
2025-10-12 10:15:30 - tokenETL - INFO - Tables to load: ['patient', 'hospitalization', 'adt', 'hospital_diagnosis']

2025-10-12 10:15:30 - tokenETL - INFO - Loading patient table...
2025-10-12 10:15:31 - tokenETL - INFO -   ✓ Loaded patient: 10,000 rows × 4 columns

2025-10-12 10:15:31 - tokenETL - INFO - Loading hospitalization table...
2025-10-12 10:15:32 - tokenETL - INFO -   ✓ Loaded hospitalization: 15,000 rows × 6 columns

...

✅ Pipeline completed successfully!
Loaded 4 tables

📊 Table Shapes:
  • patient: (10000, 4)
  • hospitalization: (15000, 6)
  • adt: (50000, 4)
  • hospital_diagnosis: (100000, 4)
```

## Output Structure

After running the pipeline, your output directory will contain:

```
output_dir/
├── tokenETL.log                    # Complete execution log
└── intermediate_tables/             # Loaded tables in parquet format
    ├── patient.parquet
    ├── hospitalization.parquet
    ├── adt.parquet
    └── hospital_diagnosis.parquet
```

## How It Works

### 1. **Configuration Loading**
- Reads `clif_config.json` for data location and settings
- Validates required fields: site, data_directory, filetype, timezone, output_dir
- Reads `config/token_config.yaml` for table definitions

### 2. **Logger Setup**
- Creates logger that writes to both console and log file
- Log file: `{output_dir}/tokenETL.log`
- Captures all loading steps, errors, and statistics

### 3. **Table Loading**
- Uses `clifpy.tables` objects directly:
  ```python
  from clifpy.tables import Patient, Hospitalization, Adt, HospitalDiagnosis

  table_obj = Patient.from_file(
      config_path='clif_config.json',
      columns=['patient_id', 'sex_category', ...]
  )
  df = table_obj.df  # pandas DataFrame
  ```

### 4. **Data Storage**
- Keeps data as **pandas DataFrames** (no Polars conversion)
- Saves intermediate tables as **Parquet files** for efficiency

## Architecture

### Differences from Previous Version

| Feature | Old (cliffm) | New (tokenETL) |
|---------|-------------|----------------|
| Table loading | ClifOrchestrator | clifpy table objects |
| Configuration | vocab_config YAML | token_config YAML |
| Data format | Polars DataFrames | Pandas DataFrames |
| Vocab config | mCIDE FM v1/v2 | Simple table/column list |
| Logging | Console only | Console + file |

### Table Class Mapping

The script maps table names to `clifpy.tables` classes:

```python
TABLE_MAP = {
    'patient': Patient,
    'hospitalization': Hospitalization,
    'adt': Adt,
    'hospital_diagnosis': HospitalDiagnosis
}
```

## Next Steps

This is **Step 1: Cohort Loading** in the tokenization pipeline.

**Future steps:**
1. ✅ Load intermediate tables (current)
2. ⏭️ Process and tokenize data
3. ⏭️ Generate training sequences
4. ⏭️ Export for foundation model training

## Troubleshooting

### Missing output_dir in config
```
ValueError: Required keys missing from clifpy config: ['output_dir']
```
**Solution:** Add `"output_dir": "/path/to/output"` to your `clif_config.json`

### Table not found
```
ValueError: Unknown table: xyz. Available: ['patient', 'hospitalization', 'adt', 'hospital_diagnosis']
```
**Solution:** Check table name spelling in `config/token_config.yaml`. Must match one of the available tables.

### Import errors
```
ModuleNotFoundError: No module named 'clifpy'
```
**Solution:** Ensure clifpy is installed: `uv sync` or check your environment

## Development

To add more tables:

1. Add table to `config/token_config.yaml`:
   ```yaml
   tables:
     vitals:
       columns:
         - hospitalization_id
         - recorded_dttm
         - vital_category
         - vital_value
   ```

2. Add import and mapping in `main.py`:
   ```python
   from clifpy.tables import ..., Vitals

   TABLE_MAP = {
       ...,
       'vitals': Vitals
   }
   ```

## Support

For issues or questions, please refer to the main CLIFATRON documentation or contact the CLIF Consortium.
