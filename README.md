# Low-Weight Errors in Bivariate Bicycle Codes

A Python framework for simulating and analyzing quantum error correction using Bivariate Bicycle (BB) codes with focus on low-weight error patterns and decoder performance.

Accompanies the research paper _Low-weight quantum syndrome errors in belief propagation decoding_: https://arxiv.org/abs/2603.19126

## References

This code is based on research on Bivariate Bicycle codes:
- Original BB codes paper: https://arxiv.org/abs/2308.07915
- Source repository: https://github.com/sbravyi/BivariateBicycleCodes
- Previous version of this repository, for the earlier paper: https://github.com/haggaila/bb-decoding
- Feedforward and state-dependent errors (including leakage) paper: https://arxiv.org/abs/2504.13083, https://doi.org/10.1103/ght4-yqmb

## License

This code is licensed under the Apache License, Version 2.0. See LICENSE file for details.

If you use this code in your research, please cite the original BB codes paper and repo pointed to above, and the current paper and repo.

## Overview

This repository provides tools for:
- **Quantum Circuit Simulation**: Simulate noisy quantum circuits with various error models including state-dependent errors, leakage, and readout errors
- **Decoder Construction**: Generate parity check matrices and decoder data for BB codes
- **Error Correction Decoding**: Decode syndromes using BP-OSD or Relay BP decoders
- **Low-Weight Error Analysis**: Identify and test decoder performance on specific low-weight fault patterns
- **Decoder Amendment**: Improve decoder performance by adding columns for problematic error patterns

## Project Structure

```
low-weight-errors/
├── bb_decoding/              # Core simulation and decoding modules
│   ├── circuit_simulation.py      # Quantum circuit simulation with noise
│   ├── database_utils.py          # Data persistence and database management
│   ├── decoder_data_setup.py      # BB code and decoder construction
│   ├── logical_simulation.py      # Monte Carlo simulation and decoding
│   ├── noise_model.py             # Noise model definitions
│   ├── setup-decoder.py           # Script: Generate decoder data
│   ├── setup-amended-decoder.py   # Script: Amend decoder with fault patterns
│   ├── run-stochastic-simulation.py    # Script: Run Monte Carlo simulations
│   ├── run-detector-simulation.py      # Script: Test specific weight-4 fault patterns
│   ├── run-randomized-simulation.py    # Script: Test randomized weight-5 fault patterns
│   ├── decoder_noise_model.yaml   # Noise model for decoder construction
│   └── simulation_noise_model.yaml     # Noise model for simulations
├── bb_output/                # Output directory (created automatically)
│   ├── decoders/            # Saved decoder data (.pkl files)
│   ├── simulations/         # Saved simulation results (.pkl files)
│   ├── figures/             # Generated plots and figures
│   └── *.database.csv       # CSV databases for tracking experiments
├── paper_plots/            # Plotting and analysis scripts
└── requirements.txt          # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd low-weight-errors
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   
   # On Windows:
   .venv\Scripts\activate
   
   # On Linux/Mac:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start Guide

Get up and running with a simple example in 3 steps:

### Step 1: Setup Environment
```bash
# Clone and setup
git clone <repository-url>
cd low-weight-errors
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Generate a Decoder

First edit parameters in the file and in the noise model yaml, then run:
```bash
# Create decoder for [[144, 12, 12]] BB code
python bb_decoding/setup-decoder.py
```

This will:
- Use the default [[144, 12, 12]] code with only 4 noisy syndrome cycles
- Apply the noise model from `bb_decoding/decoder_noise_model.yaml`
- Save decoder to `bb_output/decoders/decoder.<uuid>.pkl`
- Print the decoder UUID (copy this for next step)

### Step 3: Run a Simulation
```bash
# Edit bb_decoding/run-stochastic-simulation.py
# Set s_decoder_data_id to the UUID from Step 2
python bb_decoding/run-stochastic-simulation.py
```

This will:
- Run 100,000 Monte Carlo trials (note that for this usage, it is necessary to add one final noiseless cycle to the decoder definitions)
- Use Relay-BP decoder
- Save results to `bb_output/simulations/simulation.<uuid>.pkl`
- Print logical error rate and statistics

### View Results
Check the output databases:
```bash
# View decoder metadata
cat bb_output/decoder.database.csv

# View simulation results
cat bb_output/simulation.database.csv
```

---

## Usage

### Working Directory

**Important:** All scripts should be run from the repository root directory:
```bash
cd /path/to/low-weight-errors
```

### Basic Workflow

#### 1. Generate Decoder Data

Create decoder data for a specific BB code and noise model:

```bash
python bb_decoding/setup-decoder.py
```

**Configuration** (edit `setup-decoder.py`):
- `code_name`: Choose from "72.12.6", "144.12.12", "288.12.18", "784.24.24"
- `n_cycles`: Number of syndrome measurement cycles
- `fnc`: Final noiseless cycles for validation
- `s_noise_model_filename`: Path to noise model YAML file

**Output:**
- Decoder data saved to `bb_output/decoders/decoder.<uuid>.pkl`
- Metadata registered in `bb_output/decoder.database.csv`

#### 2. Run Monte Carlo Simulation

Simulate noisy circuits and decode syndromes:

```bash
python bb_decoding/run-stochastic-simulation.py
```

**Configuration** (edit `run-stochastic-simulation.py`):
- `s_decoder_data_id`: UUID of decoder to use (from step 1)
- `n_shots`: Number of Monte Carlo trials
- `relay_decoder`: Decoder type ("" for BP-OSD, "RelayDecoderF64" for Relay BP)
- `s_noise_model_filename`: Path to simulation noise model

**Output:**
- Simulation results saved to `bb_output/simulations/simulation.<uuid>.pkl`
- Summary statistics in `bb_output/simulation.database.csv`

#### 3. Analyze Low-Weight Errors (Optional)

Test decoder on specific fault patterns:

```bash
python bb_decoding/run-detector-simulation.py
```

**Configuration:**
- `s_decoder_data_id`: Decoder UUID
- `weight`: Fault weight to test (only 4 supported)
- `s_logical`: Error type ("X" or "Z")

#### 4. Amend Decoder (Optional)

Improve decoder by adding problematic fault patterns:

```bash
python bb_decoding/setup-amended-decoder.py
```

**Configuration:**
- `s_decoder_id`: Original decoder UUID
- `amend_fraction`: Fraction of fault patterns to add
- `b_amend_pairs`: Add fault pairs vs. complete faults

## Configuration Files

### Noise Model YAML

Noise models define error rates for quantum operations. Two example noise model files are provided:

- **`bb_decoding/decoder_noise_model.yaml`**: Noise model used for decoder construction
- **`bb_decoding/simulation_noise_model.yaml`**: Noise model used for circuit simulations

These YAML files specify error rates for various operations (preparation, gates, measurement) and optional state-dependent effects (leakage, bias, etc.). You can modify these files or create new ones to test different noise scenarios.

## Key Concepts

### Bivariate Bicycle Codes

BB codes are a family of quantum LDPC codes defined by polynomial parameters. Supported codes:
- **[[72, 12, 6]]**: Small code for testing
- **[[144, 12, 12]]**: Medium code, good performance
- **[[288, 12, 18]]**: Larger code, higher distance
- **[[784, 24, 24]]**: Largeset code

### Decoder Types

1. **BP-OSD**: Belief Propagation with Ordered Statistics Decoding
2. **Relay BP**: Enhanced BP with multiple randomized ensembles
   - Better, faster performance
   - Configurable number of relay sets and stopping criteria

### Low-Weight Error Analysis

The framework can systematically generate and test weight-4 error patterns that may escape the decoder. These patterns are characterized by:
- **Shared columns**: Faults triggering multiple detectors
- **Cancellations**: Detector pairs where faults cancel (XOR to zero)

Decoders can be "amended" by adding columns for these problematic patterns.

## Output Files

### Decoder Database (`decoder.database.csv`)
Tracks all generated decoders with metadata:
- Code parameters (n, k, d)
- Noise model parameters
- Number of cycles
- Amendment information (if applicable)

### Simulation Database (`simulation.database.csv`)
Records simulation results:
- Logical error rates
- BP convergence statistics
- Decoder performance metrics
- Timing information

### Pickle Files
Binary files containing complete data structures:
- `decoder.*.pkl`: Full decoder data (matrices, circuits, fault maps)
- `simulation.*.pkl`: Detailed simulation results (per-shot data)

## Visualization - scripts used to generate the accompanying plots

Plotting scripts in `paper_plots/` directory analyze results:
- `plot-amended-decoding.py`: Compare original vs. amended decoders
- `plot-bp-escape.py`: Analyze decoder escape patterns
- `plot-bp-histograms.py`: BP iteration distributions
- `plot-decoding-bitmaps.py`: Visualize syndrome patterns
- `plot-detectors-analysis.py`: Detector statistics
- `plot-randomized-errors.py`: Randomized weight-5 fault analysis
