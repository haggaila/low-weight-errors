# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Run decoder testing on specific low-weight fault patterns.

This script:
1. Generates weight-4 fault patterns with specific cancellation properties
2. Tests decoder performance on these patterns without circuit simulation
3. Compares performance between original and amended decoders
4. Saves results to identify decoder escape patterns

Used to analyze decoder weaknesses and validate decoder amendments.
"""

import uuid
from bb_decoding.logical_simulation import (
    simulate_detector_faults,
    choose_detector_faults_weight_4,
)
from database_utils import (
    load_decoder_data,
    save_simulation_data,
    S_DETECTOR_DB_FILENAME,
)

# Configuration
s_decoder_data_id = "25536da4ba354c8a99aed7db5373dc3b"  # Original decoder ID
s_amended_decoder_data_id = "74b0e730a19e4e1d8312d895da7bf1f0"  # Amended decoder ID (or "" for none)
s_name = "all errors, 5000 iterations, amended"  # Simulation name

relay_decoder = "RelayDecoderF64"  # Decoder type
weight = 4  # Fault weight to test
s_logical = "X"  # Error type: "X" or "Z"
rand_seed = 16217  # Random seed
n_fault_repeat = 1  # Number of times to repeat each fault pattern

# Simulation parameters
simulation_input = {
    "partial_decoding": s_logical,
    "s_logical": s_logical,
    "weight": weight,
    "rand_seed": rand_seed,
    "relay_decoder": relay_decoder,
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
            "bp_max_iterations": 5000,  # Maximum number of iterations for the BP decoder
            "osd_method": "osd_cs",  # The OSD method. Choose from "osd_e", "osd_cs", "osd0"
            "osd_order": 10,  # The osd search depth
            "ms_scaling_factor": 0,  # min-sum scaling factor. If 0, a variable-scaling factor is used
        }
    )
elif relay_decoder == "MinSumBPDecoderF32":
    # Min-sum BP decoder configuration
    simulation_input.update(
        {
            "bp_max_iterations": 10000,
            "gamma0": 0.2,  # Uniform memory weight for the first ensemble
        }
    )
else:
    # Relay BP decoder configuration
    simulation_input.update(
        {
            "gamma0": 0.125,  # Uniform memory weight for the first ensemble (Relay-BP)
            "pre_iter": 0,  # Max BP iterations for the first ensemble (Relay-BP)
            "set_max_iter": 25,  # Max BP iterations per relay ensemble (Relay-BP)
            "num_sets": 200,  # Number of relay ensemble elements (Relay-BP)
            "gamma_dist_interval": (
                -0.24,
                0.66,
            ),  # range for disordered memory weight (Relay-BP)
            "stop_nconv": 1,  # Number of relay solutions to find before stopping (Relay-BP)
        }
    )

# Load decoder data
decoder_data = load_decoder_data(s_decoder_data_id)
amended_decoder_data = (
    load_decoder_data(s_amended_decoder_data_id)
    if s_amended_decoder_data_id != ""
    else decoder_data
)

# Generate weight-4 fault patterns
(
    faults,
    fault_pairs,
    faults_to_pair_1,
    pair_shared_cols_count_matrix,
    faults_canceled_total_checks_count,
    faults_original_pair_checks_count,
    faults_canceled_pair_checks_count,
    faults_original_weights,
    faults_weight,
) = choose_detector_faults_weight_4(
    decoder_data,
    s_logical,
    n_fault_repeat=n_fault_repeat,
    pair_0_list=72,
    filter_pairs_by_shared_cols_count=[8],
    filter_syndromes_by_total_canceled_count=[8],
    filter_syndromes_by_pair_canceled_counts=[[2, 2]],
)

# Run simulation on fault patterns
bp_iterations_x_rep = []
z_fault_column_lists = faults if s_logical == "Z" else None
x_fault_column_lists = faults if s_logical == "X" else None
simulation_results, simulation_summary = simulate_detector_faults(
    simulation_input, amended_decoder_data, z_fault_column_lists, x_fault_column_lists
)
simulation_results.update(
    {
        "faults_canceled_total_checks_count": faults_canceled_total_checks_count,
        "faults_original_pair_checks_count": faults_original_pair_checks_count,
        "faults_canceled_pair_checks_count": faults_canceled_pair_checks_count,
        "faults_original_weights": faults_original_weights,
        "faults_weight": faults_weight,
    }
)

# Save results to database
db_line = {
    "code_name": decoder_data["code_name"],
    "n_cycles": decoder_data["n_cycles"],
    "fnc": decoder_data["fnc"],
    "unique_id": uuid.uuid4().hex,
    "name": s_name,
    "decoder_id": decoder_data["unique_id"],
    "amended_decoder_id": (
        amended_decoder_data["unique_id"] if s_amended_decoder_data_id else ""
    ),
    "b_amend_pairs": (
        amended_decoder_data["b_amend_pairs"] if s_amended_decoder_data_id else ""
    ),
    "amend_fraction": (
        amended_decoder_data["amend_fraction"] if s_amended_decoder_data_id else 0.0
    ),
    "n_fault_repeat": n_fault_repeat,
}
db_line.update(simulation_input)
db_line.update(simulation_summary)

save_simulation_data(db_line, simulation_results, S_DETECTOR_DB_FILENAME)
