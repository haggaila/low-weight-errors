# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# This code is based on the sources at https://github.com/sbravyi/BivariateBicycleCodes
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Noise model definitions for quantum error correction simulations.

This module defines the NoiseModel dataclass that encapsulates various error rates
and noise parameters, including gate errors, measurement errors, leakage, and
state-dependent effects. Supports loading from YAML files and generating spatially
non-uniform error rates from statistical distributions.
"""

from __future__ import annotations
import os
from typing import Dict, Optional, Sequence
import numpy as np
import yaml
from dataclasses import dataclass


@dataclass
class NoiseModel:
    """
    Dataclass representing the circuit-level noise model.
    
    Defines error rates for various gate operations including
    preparation, idle, CNOT, measurement errors, as well as effects like
    leakage, seepage, and state-dependent readout bias.
    """
    
    prep: float = 0.0
    """Qubit preparation (init) error probability."""

    id_s: float = 0.0
    """Short-duration (parallel to CNOT gate) idle-qubit error probability."""

    id_l: float = 0.0
    """Long-duration (parallel to readout) idle-qubit error probability."""

    cnot: float = 0.0
    """CNOT gate error probability."""

    meas: float = 0.0
    """Readout error probability."""

    bias: float = None
    """State-dependent readout bias, i.e. p(0|1) - p(1|0)."""

    flip: float = None
    """X-gate error probability."""

    leak: float = None
    """Probability of a check qubit to leak during measurement if it is in the state |1>."""

    seep: float = None
    """Probability of a leaked qubit to return to the state |1> during measurement."""

    back: float = None
    """Backaction probability of a leaked check qubit to scramble a data qubit during a CNOT."""

    corr: float = None
    """Probability of a preparation error correlated with (and following) a readout error."""

    b_error_before_gate: bool = None
    """If ture, error Paulis are inserted before the CNOT gates."""

    distributions: dict = None
    """Definition of device-wide distributions from which to draw local error rates per gate."""

    local_error_rates: dict = None
    """Local error probabilities, with key being the gate and qubits from the code circuit."""

    cycles: Dict = None
    """An optional dictionary of error rates overriding the defined in particular cycles."""

    metadata: object = None
    """An optional arbitrary field, ignored."""

    @property
    def is_error_before_gate(self) -> bool:
        """Returns True if error Paulis are inserted before the CNOT gates."""
        return self.b_error_before_gate is not None and self.b_error_before_gate

    @property
    def is_biased(self) -> bool:
        """Returns True if there are qubits with a biased (state-dependent) readout error."""
        return self.bias != 0.0

    @property
    def is_leaky(self) -> bool:
        """Returns True if there are qubits with a (state-dependent) leakage error."""
        return self.leak != 0.0

    @property
    def is_state_dependent(self) -> bool:
        """Returns True if the noise model is state dependent."""
        return self.is_biased or self.is_leaky

    @property
    def is_local(self) -> bool:
        """Returns True if the noise model has some local qubit- and gate-dependent error rates."""
        return self.local_error_rates is not None and len(self.local_error_rates) > 0

    @staticmethod
    def from_file(s_noise_file: str) -> NoiseModel:
        """
        Load a noise model from a YAML configuration file.
        
        Args:
            s_noise_file: Path to the YAML file containing noise parameters.
            
        Returns:
            NoiseModel instance with parameters loaded from file.
        """
        if s_noise_file == "":
            noise_model = NoiseModel()
        elif os.path.isfile(s_noise_file):
            with open(s_noise_file, "r") as fin:
                noise_file = yaml.full_load(fin)
            noise_model = NoiseModel(**noise_file)
        else:
            raise Exception(f"Noise model file {s_noise_file} not found.")
        print(f"Noise model loaded from file {s_noise_file}.")
        return noise_model

    @staticmethod
    def draw_from_distribution(rng: np.random.Generator, dist, shape) -> np.ndarray:
        """
        Draw random error rates from a specified probability distribution.
        
        Helper function for generating spatially non-uniform error rates across a device.

        Args:
            rng: A numpy.random.Generator object.
            dist: A tuple/list defining the distribution. The first entry denotes its name;
                "uniform" - the second and third tuple entries give the interval boundaries.
                "normal" - the second and third tuple entries give the mean and stddev.
                "exponential" - the second tuple entry gives the scale parameter (mean=variance).
                "samples" - all further entries beyond the first are values to be sampled from.
                    In this case the shape parameter is assumed to be an integer (for a 1D array).
                Any drawn probability value is truncated to lie within zero and one.
            shape: The numpy shape (size) of returned array (either an int or a tuple of ints).
                In the case of samples it is assumed to be an integer (for a 1D array).

        Returns:
            Array of random values clipped to [0, 1] range.

        Raises:
            Exception: If distribution name is not recognized.
        """
        dist_type = dist[0]
        dist_params = dist[1:]
        if dist_type == "uniform":
            arr = rng.uniform(dist_params[0], dist_params[1], shape)
        elif dist_type == "normal":
            arr = rng.normal(dist_params[0], dist_params[1], shape)
        elif dist_type == "exponential":
            arr = rng.exponential(dist_params[0], shape)
        elif dist_type == "samples":
            n_samples = len(dist_params)
            random_samples = rng.integers(n_samples, size=shape)
            # Draw random indices into the samples array
            arr = np.asarray(dist_params)[random_samples]
        else:
            raise Exception(f"Distribution name is unknown: {dist_type}.")
        arr = np.maximum(arr, 0.0)
        arr = np.minimum(arr, 1.0)
        return arr

    @staticmethod
    def gate_errors_from_distributions(
        rng: np.random.Generator,
        code: Dict,
        gate_error_distributions: dict,
    ) -> dict:
        """Populates gate error probabilities from indicated distributions per gate type.

        This function can be used to generate a simple extension of the noise model that
        allows breaking the spatial uniformity of the error rates, by drawing different local
        error rates (of each qubit and CNOT edge) from a device-wide random distribution
        (normal, exponential or uniform) described using a few hyperparameters.
        Args:
            rng: A numpy.random.Generator object.
            code: A dict with the BB code data to use for qubit properties.
            gate_error_distributions: A dictionary keyed by small-caps names of the basic
                supported gate types: "prep", "id_s", "id_l", "cnot", "meas". Each entry in the
                dictionary is a tuple defining the distribution, in which he first entry is a
                string denoting its name;
                    "uniform" - the second and third tuple entries give the interval boundaries.
                    "normal" - the second and third tuple entries give the mean and stddev.
                    "exponential" - the second tuple entry gives the scale parameter (mean=var).
                Any drawn probability value is truncated to lie within zero and one.
        Returns:
            A dictionary with gate error probabilities suitable for the noise model.

        Raises:
            Exception: If a distribution name is unknown.
        """
        err_rates = {}
        for s_err_type, s_dist_type in zip(["P", "M"], ["prep", "meas"]):
            dist: Optional[Sequence] = gate_error_distributions.get(s_dist_type, None)
            if dist is not None:
                for suffix, qubits in zip(
                    ["X", "Z"], [code["x_checks"], code["z_checks"]]
                ):
                    s_gate = s_err_type + suffix
                    n_qubits = len(qubits)
                    p_errors = NoiseModel.draw_from_distribution(rng, dist, n_qubits)
                    mus = None
                    for i_qubit in range(n_qubits):
                        err_rates[(s_gate, qubits[i_qubit])] = p_errors[i_qubit]
                        # Each local error on a check qubit is defined just by the gate and qubit,
                        # and the information in the dictionary is simply the flip rate.

        for s_err_type in ["ID_S", "ID_L"]:
            dist = gate_error_distributions.get(s_err_type.lower(), None)
            if dist is not None:
                s_gate = s_err_type
                qubits = code["data_qubits"]
                n_qubits = len(qubits)
                p_errors = NoiseModel.draw_from_distribution(rng, dist, n_qubits)
                # p_errors now stores the error probability of the idle gate for all qubits.
                pauli_rates = None
                for i_qubit in range(n_qubits):
                    err_rates[(s_gate, qubits[i_qubit])] = p_errors[i_qubit]
                    # err_rates will store just the probability of error for each qubit's idle gate

        n_neighbors = code["n_neighbors"]
        # The number of neighbors, used below for efficient array pre-allocations and iterations
        s_err_type = "CX"
        s_dist_type = "cnot"
        dist = gate_error_distributions.get(s_dist_type, None)
        b_dist = dist is not None
        if b_dist:
            for prefix, qubits in zip(["X", "Z"], [code["x_checks"], code["z_checks"]]):
                s_gate = s_err_type
                n_qubits = len(qubits)
                p_errors = NoiseModel.draw_from_distribution(
                    rng, dist, (n_qubits, n_neighbors)
                )
                for i_qubit in range(n_qubits):
                    check = qubits[i_qubit]
                    for i_neighbor in range(n_neighbors):
                        neighbor = code["nbs"][(qubits[i_qubit], i_neighbor)]
                        key_tuple = (
                            (s_gate, check, neighbor)
                            if prefix == "X"
                            else (s_gate, neighbor, check)
                        )
                        err_rates[key_tuple] = p_errors[i_qubit, i_neighbor]
                        # err_rates will store just the probability of error for each edge gate
        return err_rates
