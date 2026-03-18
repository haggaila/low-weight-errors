# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Run stochastic Monte Carlo simulation of quantum error correction with decoding.

This script:
1. Loads decoder data and noise model from files
2. Runs Monte Carlo simulation with noisy circuits
3. Decodes syndromes using BP-OSD or Relay BP decoder
4. Computes logical error rates and saves results to database

Configure decoder ID, noise model, and simulation parameters before running.
"""

import uuid
import dataclasses

from bb_decoding.noise_model import NoiseModel
from bb_decoding.logical_simulation import simulate_decoding
from bb_decoding.database_utils import (
    load_decoder_data,
    save_simulation_data,
    S_SIMULATION_DB_FILENAME,
)

# Simulation configuration
s_decoder_data_id = ""  # ID of decoder to use (load from database)
decoder_data = load_decoder_data(s_decoder_data_id)

n_shots = 100000  # Number of Monte Carlo trials (shots)
relay_decoder = "RelayDecoderF64"  # Decoder type:
# "" (BP-OSD), "RelayDecoderF64", "RelayDecoderI64", "MinSumBPDecoderF32"
num_sets = 300  # Number of relay ensemble sets/legs (for Relay BP)
stop_nconv = 5  # Number of relay solutions to find before stopping
rand_seed = 3222  # Random seed for reproducibility
partial_decoding = ""  # "" for both X and Z, or "x"/"z" for partial decoding
b_readout_flip = False  # Enable readout flip protocol
b_initial_state = False  # Start from random initial state
b_init_leaked = False  # Initialize with leaked qubit population
delta = 0.43 if b_readout_flip else 0.0  # Phenomenological constant for leakage
s_name = decoder_data["name"] + (
    f", R.{num_sets}.{stop_nconv}" if relay_decoder else ""
)

# Load decoder data and noise model
s_noise_model_filename = "./bb_decoding/simulation_noise_model.yaml"
noise_model = NoiseModel.from_file(s_noise_model_filename)

# Calculate initial leaked population if needed
leaked_population = 0.0
if noise_model.is_leaky and b_init_leaked:
    leaked_population = 0.5 * (1 - 2 * delta) * noise_model.leak / noise_model.seep

# Simulation parameters
simulation_input = {
    "partial_decoding": partial_decoding,
    "b_readout_flip": b_readout_flip,
    "b_initial_state": b_initial_state,
    "leaked_population": leaked_population,
    "n_shots": n_shots,
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
            "bp_max_iterations": 100,
            "gamma0": 0.5,  # Uniform memory weight for the first ensemble
        }
    )
else:
    # Relay BP decoder configuration
    simulation_input.update(
        {
            "gamma0": 0.125,  # Uniform memory weight for the first ensemble (Relay-BP)
            "pre_iter": 80,  # Max BP iterations for the first ensemble (Relay-BP)
            "set_max_iter": 60,  # Max BP iterations per relay ensemble (Relay-BP)
            "num_sets": num_sets,  # Number of relay ensemble elements (Relay-BP)
            "gamma_dist_interval": (
                -0.24,
                0.66,
            ),  # range for disordered memory weight (Relay-BP)
            "stop_nconv": stop_nconv,  # Number of relay solutions to find before stopping (Relay-BP)
        }
    )

# Run simulation
print(f"Running simulation: {s_name}.")
simulation_results, simulation_summary = simulate_decoding(
    simulation_input, decoder_data, noise_model
)

# Save results to database
db_line = {
    "code_name": decoder_data["code_name"],
    "n_cycles": decoder_data["n_cycles"],
    "fnc": decoder_data["fnc"],
    "b_readout_flip": b_readout_flip,
    "unique_id": uuid.uuid4().hex,
    "name": s_name,
    "decoder_id": decoder_data["unique_id"],
}
db_line.update(dataclasses.asdict(noise_model))
db_line.update(simulation_summary)
simulation_input.pop("b_readout_flip")
db_line.update(simulation_input)
decoder_noise_model = dataclasses.asdict(decoder_data["noise_model"])
decoder_noise_model.pop("local_error_rates")
db_line["decoder_noise_model"] = decoder_noise_model

save_simulation_data(db_line, simulation_results, S_SIMULATION_DB_FILENAME)
