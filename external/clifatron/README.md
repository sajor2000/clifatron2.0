# CLIFATRON

[![PyPI version](https://badge.fury.io/py/clifatron.svg)](https://badge.fury.io/py/clifatron) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**CLIFATRON** (Clinical Longitudinal ICU Forecasting with Attention-based TRansformer for Outcome and Next-event prediction) is a comprehensive Python library for processing CLIF format of Electronic Health Record (EHR) data and training transformer-based models for clinical prediction tasks.

Built on the Common Longitudinal ICU data Format (CLIF) and minimum Common ICU Data Elements (mCIDE) ontology standards, CLIFATRON provides a standardized framework for:

-   🏥 **Clinical Data Processing**: Tokenization and preprocessing of ICU data
-   🤖 **Transformer Models**: Attention-based models for outcome and next-event prediction\
-   📊 **mCIDE Vocabulary**: Standardized vocabulary following minimum Common ICU Data Elements
-   🔄 **Sentence Generation**: Converting clinical data into transformer-ready sequences
-   🎯 **Multi-center Research**: Facilitating federated critical care research

## Key Features

-   **Standardized Vocabulary**: All models use the mCIDE Foundation Model vocabulary (`mcide_fm_v1.yaml`)
-   **Transformer Architecture**: Built for attention-based clinical prediction models
-   **CLIF Format Support**: Native support for Common Longitudinal ICU data Format
-   **Extensible Design**: Easy to extend with custom vocabularies for specialized use cases
-   **Research Ready**: Follows FAIR data principles and NIH Data Management guidelines

## Installation

``` bash
pip install clifatron
```

For development installation with optional dependencies:

``` bash
pip install clifatron[ml,viz,dev]
```

## Quick Start

### Basic Usage

``` python
from clifatron import ClifFM

# Initialize CLIFATRON with your CLIF data directory
clif_fm = ClifFM(
    data_dir="/path/to/your/clif_data",
    file_format='parquet'  # Recommended format
)

# Process data using mCIDE vocabulary
print("🔄 Processing clinical data with mCIDE vocabulary...")
clif_fm.dataprep(vocab_config='mcide_fm_v1', generate_sentences=False)
print("✅ Data processing completed")

# Generate transformer-ready sentences
print("🔤 Generating clinical sentences...")
sentences_df = clif_fm.sentences_assembler(debug=True, output=True)

# Display results
print(f"Generated {len(sentences_df)} clinical sentence sequences")
print("Sample sequences:")
print(sentences_df.head())
```

### Custom Vocabulary Configuration

While CLIFATRON's released models work exclusively with the official mCIDE vocabulary, you can create custom configurations for your specific research needs:

``` python
# Note: Custom vocabularies are for specialized research only
# Released models will not work with custom configurations
clif_fm.dataprep(vocab_config='your_custom_config', generate_sentences=False)
```

## Vocabulary Standards

CLIFATRON uses the **mCIDE (minimum Common ICU Data Elements)** vocabulary system:

-   **Standardized**: Precisely defined clinical entities with limited permissible values
-   **Interoperable**: Consistent data representation across multiple studies\
-   **Extensible**: Future vocabulary versions will support additional data elements
-   **Research-Grade**: Designed for transformer-based clinical prediction models

All `*_category` variables in the vocabulary are Common Data Elements (CDEs). The current vocabulary version is defined in:

```         
clifatron/vocab_config/mcide_fm_v1.yaml
```

## Clinical Data Elements

CLIFATRON processes these key clinical data types:

-   **Patient Demographics**: Age, sex, admission/discharge information
-   **Vital Signs**: Heart rate, blood pressure, temperature, SpO2, respiratory rate
-   **Laboratory Values**: Complete blood count, chemistry panel, arterial blood gas
-   **Medications**: Continuous infusions (vasopressors, sedatives, analgesics)
-   **Assessments**: GCS, RASS, and other clinical scores
-   **Location Tracking**: ICU, ward, step-down, procedural areas
-   **Respiratory Support**: Ventilator settings, oxygen delivery devices

## Tokenization and Narrative Assembly

CLIFATRON includes a comprehensive tokenization pipeline (`tokenETL`) that converts raw CLIF data into transformer-ready narrative sequences. The pipeline runs in two sequential steps:

### Prerequisites

Create a `clif_config.json` file in the project root directory:

``` json
{
  "site": "your_site_name",
  "data_directory": "/path/to/your/clif_data",
  "filetype": "parquet",
  "timezone": "US/Central",
  "output_dir": "/path/to/output"
}
```

### Step 1: Tokenization

Run the tokenization pipeline to process all clinical data domains:

``` bash
uv run tokenETL/main.py --config clif_config.json
```

**What it does:** - **Phase 1-4**: Build cohort and tokenize vitals, labs, and assessments - **Phase 5**: Tokenize medication administrations (continuous infusions) - **Phase 6**: Tokenize ADT (location transfers) - **Phase 7**: Tokenize respiratory support (17+ parameters with intervals) - **Phase 8**: Tokenize CRRT and ECMO/MCS therapies - **Phase 9**: Add demographics and Elixhauser comorbidities

**Output:** Token tables saved as parquet files in `{output_dir}/token_tables/`: - `cohort.parquet` - Patient cohort with demographics - `vitals.parquet`, `labs.parquet`, `assessment.parquet` - Clinical measurements - `medication_admin_continuous.parquet` - Medication infusions - `adt.parquet` - Location transfers - `respiratory_support.parquet` - Ventilator settings and parameters - `crrt_therapy.parquet`, `ecmo_mcs.parquet` - Advanced therapies

### Step 2: Narrative Assembly

Combine all token tables into chronological narrative sequences:

``` bash
uv run tokenETL/assemble_narratives.py --config clif_config.json
```

**What it does:** - Loads all token tables and cohort data - Calculates day/hour markers relative to first event (not admission) - Inserts special tokens (PREV_NARRATIVE_START/END, demographics) - Sorts all events chronologically by hospitalization - Generates complete narrative sequences

**Output files** in `{output_dir}`: - `final_narrative.parquet` - Complete narrative sequences (all hospitalizations) - `narrative_token_counts.csv` - Token frequency counts by source - `example_narrative_df.csv` - Sample narrative for inspection - `token_summary_statistics.csv` - Token count distributions - `example_narrative_max_medication_coverage.txt` - Detailed example with max medications

### Typical Workflow

``` bash
# 1. Configure your data paths
vim clif_config.json

# 2. Run tokenization (may take several hours for large datasets)
uv run tokenETL/main.py --config clif_config.json

# 3. Assemble narratives
uv run tokenETL/assemble_narratives.py --config clif_config.json

# 4. Verify output
ls -lh /path/to/output/final_narrative.parquet
```

### Output Structure

Each narrative sequence contains: - **Special tokens**: PREV_NARRATIVE_START, PREV_NARRATIVE_END - **Demographics**: sex_category, age_category - **Elixhauser comorbidities**: elix\_\* tokens (if applicable) - **Temporal markers**: day_1, day_2, ..., day_30+ and hour_1, ..., hour_24 - **Clinical events**: Chronologically ordered tokens from all data domains - **Discharge**: disposition_category token

**Example sequence structure:**

```         
PREV_NARRATIVE_START
  elix_congestive_heart_failure
  elix_chronic_pulmonary
PREV_NARRATIVE_END
sex_male
age_56_65
day_1
  hour_11
    vitals_heart_rate_(100.0,110.0]
    labs_sodium_(135.0,140.0]
    medications_norepinephrine_(0.05,0.10]
  hour_12
    vitals_sbp_(110.0,120.0]
day_2
  hour_1
    transfer_to_ward
disposition_home
```

### Key Features

-   **Day calculation**: Uses first event time as day 1 baseline (supports pre-admission events)
-   **Interval-based tokens**: Numeric values binned into clinically meaningful intervals
-   **Temporal ordering**: Events sorted by event_time, then sequence_order
-   **Polars-optimized**: 10-100x faster than pandas for large datasets
-   **Memory efficient**: Processes millions of events with minimal memory footprint

## Project Roadmap

As CLIF grows and more data elements become available, CLIFATRON will evolve with:

-   📈 **Advanced Model Architectures**: Enhanced transformer variants for clinical prediction
-   🧠 **Multi-modal Integration**: Support for clinical notes, imaging, and waveform data\
-   🔬 **Specialized Models**: Disease-specific and outcome-focused prediction models
-   🌐 **Federated Learning**: Tools for multi-center collaborative model training
-   📊 **Extended Vocabularies**: Future mCIDE versions with expanded clinical elements

## Documentation

For comprehensive documentation and tutorials:

-   **Official CLIF Website**: [clif-consortium.github.io](https://clif-consortium.github.io/)
-   **CLIF Project Workflow**: [Documentation Portal](https://clif-consortium.github.io/clif-documentation/)
-   **mCIDE Standards**: [CLIF Data Standards](https://clif-icu.com)

## Contributing

We welcome contributions to CLIFATRON! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

-   Code style and standards
-   Testing requirements\
-   Documentation guidelines
-   Pull request process

## Citation

If you use CLIFATRON in your research, please cite:

``` bibtex
@software{clifatron2025,
  title={CLIFATRON: Clinical Longitudinal ICU Forecasting with Attention-based Transformer},
  author={CLIF Consortium},
  year={2024},
  url={https://github.com/clif-consortium/clifatron}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

CLIFATRON is developed by the CLIF Consortium to advance critical care research through standardized data representation and transformer-based clinical prediction models. We thank the clinical informatics and machine learning communities for their continued support.

------------------------------------------------------------------------

**Note**: CLIFATRON released models are specifically designed to work with the official mCIDE vocabulary configuration. While the library supports custom configurations for specialized research, compatibility with pre-trained models requires adherence to the standard mCIDE vocabulary format.