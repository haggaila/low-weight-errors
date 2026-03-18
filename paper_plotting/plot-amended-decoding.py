# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Plot comparison of decoder performance with different amendment fractions.

Generates plots showing:
- Mean BP iterations vs. amendment fraction
- Logical error probability vs. amendment fraction
Compares BP-OSD and Relay BP decoders.
"""

import matplotlib.pyplot as plt

from bb_decoding.database_utils import (
    query_simulations_by_name,
    generate_decoding_paths,
    S_DETECTOR_DB_FILENAME,
)

# Configuration
fontsize = 20
b_save_figures = True
s_simulations_name = "all errors, 5000 iterations, amended"

# Load simulation data
s_output_path, _, _, s_plot_path = generate_decoding_paths(0)
plt.rc("font", family="serif")
plt.rcParams["font.size"] = fontsize

sims = query_simulations_by_name(
    s_simulations_name, n_up_dirs=0, s_db_filename=S_DETECTOR_DB_FILENAME
)
sims_relay = sims.loc[sims["relay_decoder"].astype("bool")]
sims_osd = sims.loc[~sims["relay_decoder"].astype("bool")].sort_values(
    by="amend_fraction"
)
sims_flip_0 = sims_relay.loc[~sims_relay["b_amend_pairs"].astype("bool")]
bp_iterations = [
    sims_osd.bp_iterations.values,
    sims_flip_0.bp_iterations.values,
]
amended_fractions = [
    sims_osd.amend_fraction.values.astype("float"),
    sims_flip_0.amend_fraction.values.astype("float"),
]
shots_err = [
    sims_osd.bad_shots.values / sims_osd.n_shots.values,
    sims_flip_0.bad_shots.values / sims_flip_0.n_shots.values,
]

# Plot comparison
s_labels = ["BP+OSD", "Relay-BP"]
fmts = ["-s", "-.^", ":>", "-<", "+", "x", "d"]
fig, axs = plt.subplots(2, 1, figsize=(11, 7))
for i_amend in [0, 1]:
    axs[0].plot(
        amended_fractions[i_amend],
        bp_iterations[i_amend],
        fmts[i_amend],
        label=s_labels[i_amend],
    )
    axs[1].plot(
        amended_fractions[i_amend],
        shots_err[i_amend],
        fmts[i_amend],
        label=s_labels[i_amend],
    )
axs[0].set_title("(a) Mean number of BP iterations")
axs[1].set_title("(b) Logical error probability")
axs[0].set_yscale("log")
axs[0].set_ylim((4.5, 5.5e3))
axs[0].legend(fontsize=fontsize)
axs[1].legend(fontsize=fontsize)
axs[1].set_xlabel("Fraction of weight-four errors added", fontsize=fontsize + 2)
plt.tight_layout(pad=0.8)
if b_save_figures:
    s_file_name = s_plot_path + "amended"
    plt.savefig(s_file_name + ".pdf")

plt.show()
tmp = 2
