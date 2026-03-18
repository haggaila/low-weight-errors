# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Setup script to amend an existing decoder with additional low-weight fault patterns.

This script:
1. Loads an existing decoder by ID
2. Identifies weight-4 fault patterns that may escape the decoder
3. Adds selected fault patterns (or pairs) to the parity check matrix
4. Saves the amended decoder as a new entry in the database

This improves decoder performance on low-weight errors.
"""

import uuid
import itertools
import numpy as np
from bb_decoding.database_utils import (
    query_simulations_by_name,
    load_decoder_data,
    S_DECODER_DATA_DB_FILENAME,
    save_decoder_data,
)
from bb_decoding.decoder_data_setup import amend_decoder_in_place
from bb_decoding.logical_simulation import choose_detector_faults_weight_4

# Configuration
s_decoder_id = "25536da4ba354c8a99aed7db5373dc3b"  # ID of decoder to amend
amend_fraction = 1.0  # Fraction of the columns to amend
b_amend_pairs = True  # Amend fault pairs instead of individual faults
s_amend_suffix = ", amended, one cycle"  # Suffix for amended decoder name
rand_seed = 131  # Random seed for reproducibility

# Load existing decoder
decoder_data = load_decoder_data(s_decoder_id)
rng = np.random.default_rng(rand_seed)
# Generate and add amendment columns for both X and Z decoders
n_amend_columns_z, n_amend_columns_x = 0, 0
for s_decoder_type in ["z", "x"]:
    # Find weight-4 fault patterns with specific cancellation properties
    (
        faults,
        fault_pairs,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = choose_detector_faults_weight_4(
        decoder_data,
        s_decoder_type,
        pair_0_list=72,
        filter_pairs_by_shared_cols_count=[8],
        filter_syndromes_by_total_canceled_count=[8],
        filter_syndromes_by_pair_canceled_counts=[[2, 2]],
    )
    print(f"faults: {len(faults)}")
    print(f"fault_pairs: {len(fault_pairs)}")
    if b_amend_pairs:
        amend_pairs = []
        for fault in faults:
            for comb_1 in itertools.combinations(fault, 2):
                comb_1_set = set(comb_1)
                if comb_1_set not in amend_pairs:
                    amend_pairs.append(comb_1_set)
        amend_faults = rng.choice(
            amend_pairs, size=int(amend_fraction * len(amend_pairs)), replace=False
        )
        print(f"amend_pairs: {len(amend_pairs)}")
    else:
        amend_faults = rng.choice(
            faults, size=int(amend_fraction * len(faults)), replace=False
        )
    n_amend_columns = len(amend_faults)
    if s_decoder_type == "z":
        n_amend_columns_z = n_amend_columns
    else:
        n_amend_columns_x = n_amend_columns
    print(
        f"Adding {n_amend_columns} {s_decoder_type}-type "
        f"{'fault-pair' if b_amend_pairs else 'complete fault'} columns."
    )
    amend_decoder_in_place(decoder_data, s_decoder_type, amend_faults)

# Save amended decoder with new ID
df = query_simulations_by_name(
    "", n_up_dirs=0, s_db_filename=S_DECODER_DATA_DB_FILENAME
)
s_amended_decoder_id = uuid.uuid4().hex
db_line = dict(df.loc[df["unique_id"] == s_decoder_id].iloc[0])
s_name = db_line["name"] + s_amend_suffix
amend_data = {
    "unique_id": s_amended_decoder_id,
    "name": s_name,
    "parent_decoder_id": s_decoder_id,
    "amend_rand_seed": rand_seed,
    "b_amend_pairs": b_amend_pairs,
    "amend_fraction": amend_fraction,
    "n_amend_columns_z": n_amend_columns_z,
    "n_amend_columns_x": n_amend_columns_x,
}

db_line.update(amend_data)
decoder_data.update(amend_data)
save_decoder_data(db_line, decoder_data)
