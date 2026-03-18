# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Analyze detector pair relationships and weight-4 fault patterns.

Generates plots showing:
- Heatmaps of shared columns between detector pairs
- Histograms of shared column counts for X and Z errors
- Distribution of canceled checks in weight-4 faults
"""

import matplotlib.pyplot as plt
from bb_decoding.database_utils import (
    generate_decoding_paths,
    load_decoder_data,
)
from bb_decoding.logical_simulation import (
    generate_pair_shared_cols_matrix,
    choose_detector_faults_weight_4,
)

fontsize = 16
b_save_figures = True
s_decoder_id = "2f7f04a2da714395800205b55c9841d4"

decoder_data = load_decoder_data(s_decoder_id)
s_output_path, _, _, s_plot_path = generate_decoding_paths(0)
plt.rc("font", family="serif")
plt.rcParams["font.size"] = fontsize

decoder_types = ["x", "z", "z"]
n_check_groups = [4, 4, 1]
shared_cols_count = []
for s_decoder_type, n_groups in zip(decoder_types, n_check_groups):
    H_decoder = decoder_data[f"H{s_decoder_type.upper()}_decoder"]
    _, _, pair_shared_cols_count_matrix, _ = generate_pair_shared_cols_matrix(
        H_decoder, rows=range(n_groups * 72)
    )
    shared_cols_count.append(pair_shared_cols_count_matrix)
    fig, axs = plt.subplots(1, 1, figsize=(8, 6.4))
    plt.imshow(pair_shared_cols_count_matrix, interpolation="nearest", resample=False)
    plt.xlabel("Detector", fontsize=fontsize)
    plt.ylabel("Detector", fontsize=fontsize)
    plt.colorbar()
    plt.tight_layout(pad=0.8)
    if b_save_figures:
        s_file_name = s_plot_path + f"{s_decoder_type}.{n_groups}.pairs"
        plt.savefig(s_file_name + ".pdf")

fontsize = 18
plt.rcParams["font.size"] = fontsize
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
count_hist = [
    shared_cols_count[0][72, 72:144].flatten(),
    shared_cols_count[1][0, 0:72].flatten(),
]
n_bins = 9
hatches = ["\\\\\\", "///"]
n, bins, patches = plt.hist(
    count_hist,
    label=["X-type errors", "Z-type errors"],
    bins=n_bins,
    range=(-0.5, 8.5),
    align="mid",
    hatch=hatches,
)
plt.xlabel("Count of shared columns", fontsize=fontsize)
indexes = range(n_bins)
ax.set_xticks(indexes, indexes)
plt.ylabel("Frequency", fontsize=fontsize)
ax.legend(fontsize=fontsize)
plt.tight_layout(pad=1)
if b_save_figures:
    s_file_name = s_plot_path + "shared"
    plt.savefig(s_file_name + ".pdf")

count_hist = []
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
pair_0_list = [(72, 94), (0, 22)]
decoder_types = ["x", "z"]
for pair_0, s_decoder_type in zip(pair_0_list, decoder_types):
    (
        _,
        _,
        _,
        _,
        faults_canceled_total_checks_count,
        _,
        _,
        _,
        _,
    ) = choose_detector_faults_weight_4(
        decoder_data,
        s_decoder_type,
        pair_0_list=[pair_0],
        filter_pairs_by_shared_cols_count=[8],
    )
    count_hist.append(faults_canceled_total_checks_count)
n_bins = 11
hatches = ["\\\\\\", "///"]
n, bins, patches = plt.hist(
    count_hist,
    label=["X-type errors", "Z-type errors"],
    bins=n_bins,
    range=(-0.5, 10.5),
    align="mid",
    hatch=hatches,
)
plt.xlabel("Count of canceled checks", fontsize=fontsize)
indexes = range(n_bins)
ax.set_xticks(indexes, indexes)
plt.ylabel("Frequency", fontsize=fontsize)
plt.yscale("log")
ax.legend(fontsize=fontsize)
plt.tight_layout(pad=1)
if b_save_figures:
    s_file_name = s_plot_path + "canceled"
    plt.savefig(s_file_name + ".pdf")

plt.show()
tmp = 2
