# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Visualize BP decoding process as bitmap heatmaps.

Creates grayscale bitmap images showing which detectors are active during
BP iterations for both failing and successful decoding cases.
"""

import numpy as np
import matplotlib.pyplot as plt
from bb_decoding.database_utils import generate_decoding_paths

fontsize = 22
pixel_scale = 1
b_save_figures = True
iterations = 4 * 60 + 11
faults = 122
b_inv = False

plt.rc("font", family="serif")
plt.rcParams["font.size"] = fontsize
s_output_path, _, s_simulation_path, s_plot_path = generate_decoding_paths()

decoding_bits = np.load(s_output_path + "/relay_bp_data/x_decoding_0.npy")
i0 = 106 * 60
iterations_sum = np.sum(decoding_bits, axis=0)
iterations_order = np.argsort(iterations_sum)

im = np.kron(
    decoding_bits[i0 : i0 + iterations, iterations_order[-faults:-1]],
    np.ones((1, pixel_scale), dtype=decoding_bits.dtype),
)
im = np.transpose(im)
if b_inv:
    im = np.invert(im) - 254
fig, axs = plt.subplots(1, 1, figsize=(12, 6.5))
plt.imshow(
    im[::-1,] * 255,
    cmap="gray",
    vmin=0,
    vmax=255,
    interpolation="nearest",
    resample=False,
)
# plt.axis('off')
plt.tight_layout(pad=1.2)
plt.xlabel("BP iteration", fontsize=fontsize)
plt.ylabel("Fault", fontsize=fontsize)
if b_save_figures:
    s_file_name = s_plot_path + "fault" + (".inv" if b_inv else "")
    plt.savefig(s_file_name + ".pdf")

decoding_bits = np.load(s_output_path + "/relay_bp_data/x_decoding_c.npy")
i0 = 0 * 60
iterations_sum = np.sum(decoding_bits, axis=0)
iterations_order = np.argsort(iterations_sum)

im = np.kron(
    decoding_bits[i0 : i0 + iterations, iterations_order[-faults:-1]],
    np.ones((1, pixel_scale), dtype=decoding_bits.dtype),
)
im = np.transpose(im)
if b_inv:
    im = np.invert(im) - 254
fig, axs = plt.subplots(1, 1, figsize=(12, 6.5))
plt.imshow(
    im[::-1,] * 255,
    cmap="gray",
    vmin=0,
    vmax=255,
    interpolation="nearest",
    resample=False,
)
plt.tight_layout(pad=1.2)
plt.xlabel("BP iteration", fontsize=fontsize)
plt.ylabel("Fault", fontsize=fontsize)
if b_save_figures:
    s_file_name = s_plot_path + "success" + (".inv" if b_inv else "")
    plt.savefig(s_file_name + ".pdf")

plt.show()
