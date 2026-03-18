# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Run decoder testing on randomized fault patterns (weight-4 + 1 additional fault).

This script:
1. Starts with a known weight-4 fault pattern
2. Adds one additional random fault from the reduced column set
3. Tests decoder performance on these weight-5 patterns
4. Analyzes how additional faults affect decoding success

Used to study decoder robustness to slightly higher weight errors.
"""

import uuid
import numpy as np
from matplotlib import pyplot as plt

from bb_decoding.logical_simulation import simulate_detector_faults
from database_utils import (
    load_decoder_data,
    save_simulation_data,
    S_DETECTOR_DB_FILENAME,
)

# Configuration
s_decoder_data_id = "25536da4ba354c8a99aed7db5373dc3b"  # Decoder ID
relay_decoder = "RelayDecoderF64"  # Decoder type
weight = "4+1"  # Base weight-4 plus one additional fault
s_logical = "Z"  # Error type: "X" or "Z"
rand_seed = 16215  # Random seed
s_name = "weight-4 errors with one more fault"  # Simulation name
n_fault_repeat = 100  # Repetitions per fault combination

# Base weight-4 fault patterns (specific column indices)
if s_logical == "Z":
    fault_cols = [310, 313, 460, 466]
    # [457, 301, 310, 295] [163, 307, 313, 322] [328, 313, 337, 319] [313, 301, 310, 295]
    # [310, 313, 460, 466] [310, 313, 322, 460] [310, 313, 316, 322] [310, 313, 316, 466]
else:
    fault_cols = [650, 716, 509, 719]
    # Example patterns: [506, 659, 653, 719] [704, 713, 701, 719] [713, 698, 707, 798] [650, 716, 509, 719]

# Simulation parameters
simulation_input = {
    "partial_decoding": s_logical,
    "s_logical": s_logical,
    "weight": weight,
    "rand_seed": rand_seed,
    "relay_decoder": relay_decoder,
    "b_initial_state": False,
    "bp_method": "",
    "bp_max_iterations": "",
    "osd_method": "",
    "osd_order": "",
    "ms_scaling_factor": "",
    "gamma0": "",
    "pre_iter": "",
    "num_sets": "",
    "set_max_iter": "",
    "gamma_dist_interval": "",
    "stop_nconv": "",
}

# Configure decoder-specific parameters
if relay_decoder == "":
    # BP-OSD decoder configuration
    simulation_input.update(
        {
            "bp_method": "ms",  # BP method for the BP decoder
            "bp_max_iterations": 10000,  # Maximum number of iterations for the BP decoder
            "osd_method": "osd_cs",  # The OSD method. Choose from "osd_e", "osd_cs", "osd0"
            "osd_order": 10,  # The osd search depth
            "ms_scaling_factor": 0,  # min-sum scaling factor. If 0, a variable-scaling factor is used
        }
    )
elif relay_decoder == "MinSumBPDecoderF32":
    # Min-sum BP decoder configuration
    simulation_input.update(
        {
            "bp_max_iterations": 1000,
            "gamma0": 0.2,  # Uniform memory weight for the first ensemble
        }
    )
else:
    # Relay BP decoder configuration
    simulation_input.update(
        {
            "gamma0": 0.125,  # Uniform memory weight for the first ensemble (Relay-BP)
            "pre_iter": 0,  # Max BP iterations for the first ensemble (Relay-BP)
            "set_max_iter": 50,  # Max BP iterations per relay ensemble (Relay-BP)
            "num_sets": 200,  # Number of relay ensemble elements (Relay-BP)
            "gamma_dist_interval": (
                -0.24,
                0.66,
            ),  # range for disordered memory weight (Relay-BP)
            "stop_nconv": 1,  # Number of relay solutions to find before stopping (Relay-BP)
        }
    )

# Load decoder and build reduced column set
decoder_data = load_decoder_data(s_decoder_data_id)
H_decoder = decoder_data["HZ_decoder" if s_logical == "Z" else "HX_decoder"]

# Build set of columns connected to the base fault pattern
reduced_cols_set = set()
for fault_col in fault_cols:
    checks = np.where(H_decoder.getcol(fault_col).toarray().flatten())[0]
    for check in checks:
        cols = np.where(H_decoder.getrow(check).toarray().flatten())[0]
        reduced_cols_set = reduced_cols_set | set(cols)

# Randomize order of additional faults to test
rng = np.random.default_rng(rand_seed)
fault_cols_set = set(fault_cols)
reduced_cols = rng.permutation(np.asarray(list(reduced_cols_set - fault_cols_set)))
n_reduced_cols = len(reduced_cols)
decoder_cols = fault_cols.copy() + list(reduced_cols)

# Generate fault patterns (base + one additional)
simulations = []
bp_iterations = np.zeros(n_reduced_cols)
r_reduced_cols = range(n_reduced_cols)
c = None
simulation_results, simulation_summary = None, None
z_fault_column_lists = [] if s_logical == "Z" else None
x_fault_column_lists = [] if s_logical == "X" else None
for i in r_reduced_cols:
    if s_logical == "Z":
        z_fault_column_lists.extend(
            [fault_cols + ([c] if c is not None else [])] * n_fault_repeat
        )
    if s_logical == "X":
        x_fault_column_lists.extend(
            [fault_cols + ([c] if c is not None else [])] * n_fault_repeat
        )
    c = decoder_cols.pop()

# Run simulation
simulation_results, simulation_summary = simulate_detector_faults(
    simulation_input, decoder_data, z_fault_column_lists, x_fault_column_lists
)

# Save results to database
db_line = {
    "code_name": decoder_data["code_name"],
    "n_cycles": decoder_data["n_cycles"],
    "fnc": decoder_data["fnc"],
    "unique_id": uuid.uuid4().hex,
    "name": s_name,
    "decoder_id": decoder_data["unique_id"],
    "amended_decoder_id": "",
    "b_amend_pairs": False,
    "amend_fraction": 0.0,
    "n_fault_repeat": n_fault_repeat,
}
db_line.update(simulation_input)
db_line.update(simulation_summary)
save_simulation_data(db_line, simulation_results, S_DETECTOR_DB_FILENAME)
