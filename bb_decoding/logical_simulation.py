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
Logical-level simulation and decoding routines.

This module provides functions to:
- Run Monte Carlo simulations of noisy quantum circuits
- Decode syndromes using BP-OSD or Relay BP decoders
- Test decoder performance on specific fault patterns
- Analyze low-weight error escape mechanisms
"""

import copy
import itertools
import json
import numpy as np
from typing import Dict, Optional, List, Union, Tuple
from datetime import datetime
from ldpc import bposd_decoder
import relay_bp
from scipy.sparse import csr_matrix
from bb_decoding.noise_model import NoiseModel
from bb_decoding.circuit_simulation import (
    generate_noisy_circuit,
    get_detector_history,
    simulate_errors,
)


def simulate_decoding(
    simulation_input: Dict,
    decoder_data: Dict,
    noise_model: Optional[NoiseModel] = None,
) -> (Dict, Dict):
    """
    Run Monte Carlo simulation of noisy circuits with decoding.
    
    Simulates multiple shots of the quantum circuit with noise, generates syndromes,
    decodes them using BP-OSD or Relay BP, and tracks logical error rates.
    
    Args:
        simulation_input: Simulation parameters (n_shots, decoder settings, etc.).
        decoder_data: Decoder data structure with parity check matrices.
        noise_model: Noise model specifying error rates and behavior.
        
    Returns:
        Tuple of (results_dict, summary_dict) with detailed and summary statistics.
    """
    if noise_model is not None and noise_model.is_local:
        raise Exception(
            "Simulation of decoding with local (gate-dependent) error rates not implemented!"
        )

    b_readout_flip = simulation_input.get("b_readout_flip", False)
    # Whether to simulate the readout flip protocol
    leaked_population = simulation_input.get("leaked_population", 0)
    # The overall mean leaked population to assign in check qubits, for entering the decoding
    # phase with the qubits close to a leakage/seepage steady state.
    b_initial_state = simulation_input["b_initial_state"]
    # Start from a random initial state and a first ideal cycle.
    rand_seed = simulation_input.get("rand_seed", 42)
    output_step = simulation_input.get("output_step", 50)
    # Number of shots to skip between console output lines if no error occurred
    s_logical = simulation_input.get("s_logical", "")
    weight = simulation_input.get("weight", "")
    relay_decoder = simulation_input.get("relay_decoder", "")
    partial_decoding = simulation_input.get("partial_decoding", "").lower()

    n_shots = simulation_input.get("n_shots", 0)
    # Number of Monte Carlo trials (shots) in the simulation.
    HX_decoder = decoder_data["HX_decoder"]
    HZ_decoder = decoder_data["HZ_decoder"]
    lin_order = decoder_data["lin_order"]
    data_qubits = decoder_data["data_qubits"]
    data_qubit_indices = decoder_data["data_qubit_indices"]
    x_checks = decoder_data["x_checks"]
    z_checks = decoder_data["z_checks"]
    cycle = decoder_data["cycle"]
    HX = decoder_data["HX"]
    HZ = decoder_data["HZ"]
    lx = decoder_data["lx"]
    lz = decoder_data["lz"]
    z_logical_row = decoder_data["z_logical_row"]
    x_logical_row = decoder_data["x_logical_row"]
    ell = decoder_data["ell"]
    m = decoder_data["m"]
    n = decoder_data["n"]
    k = decoder_data["k"]
    n_cycles = decoder_data["n_cycles"]
    fnc = decoder_data["fnc"]
    n2 = m * ell
    noise_models = []
    cycle_repeated = []
    for i_cycle in range(n_cycles):
        cycle_noise = None
        if noise_model is not None and noise_model.cycles is not None:
            cycle_params: Optional[Dict] = noise_model.cycles.get(i_cycle, None)
            if cycle_params is not None:
                cycle_noise = copy.copy(noise_model)
                for key, val in cycle_params.items():
                    setattr(cycle_noise, key, val)
        if cycle_noise is None:
            cycle_noise = noise_model
        noise_models.append(cycle_noise)
        cycle_repeated.extend(cycle.copy())

    x_bpd, z_bpd = create_decoders(simulation_input, decoder_data)
    b_state_sim = noise_model is not None and noise_model.is_state_dependent
    rng = np.random.default_rng(rand_seed)
    bad_shots = 0
    bad_shots_z = 0
    bad_shots_x = 0
    t1 = datetime.now()
    shot_fails = [{}, {}]
    bp_non_converged = [{}, {}]
    bp_converged = np.full((n_shots, 2), 0, dtype=int)
    bp_iterations = np.full((n_shots, 2), 0, dtype=int)
    unsatisfied_fraction = np.full((n_shots, 2), 0.0)
    # The fraction of check detectors (the result of the xor in time of each check)
    # that are "unsatisfied" (not 0), averaged over all cycles. This is not currently
    # saved to the results and should be removed.
    mean_ground = np.full((n2 * n_cycles, 2), 0.0)
    mean_leaked = np.full((n2 * n_cycles, 2), 0.0)
    for i_shot in range(n_shots):
        circ = generate_noisy_circuit(cycle_repeated, noise_models, rng)
        z_initial_state = np.zeros(2 * n, dtype=int)
        x_initial_state = np.zeros(2 * n, dtype=int)
        if b_initial_state:
            z_initial_state[data_qubit_indices] = rng.binomial(1, 0.5, n)
            x_initial_state[data_qubit_indices] = rng.binomial(1, 0.5, n)
            full_circuit = cycle.copy()
            if b_state_sim:
                full_circuit += [("ON",)]
            full_circuit += circ
        else:
            full_circuit = circ.copy()
        if b_state_sim:
            full_circuit += [("OFF",)]
        for i in range(fnc):
            full_circuit += cycle

        (
            z_pre_readout_history,
            z_syndrome_history,
            z_syndrome_map,
            z_state,
            x_pre_readout_history,
            x_syndrome_history,
            x_syndrome_map,
            x_state,
        ) = simulate_errors(
            full_circuit,
            n2,
            lin_order,
            rng,
            z_initial_state,
            x_initial_state,
            noise_model,
            b_readout_flip,
            leaked_population,
        )
        if b_initial_state:
            z_state = (z_state + z_initial_state) % 2
            x_state = (x_state + x_initial_state) % 2
        if b_state_sim:
            assert len(z_pre_readout_history) == n_cycles * n2
            assert len(x_pre_readout_history) == n_cycles * n2
            mean_ground[:, 0] += np.asarray(z_pre_readout_history == 0, dtype=float)
            mean_leaked[:, 0] += np.asarray(z_pre_readout_history == 3, dtype=float)
            mean_ground[:, 1] += np.asarray(x_pre_readout_history == 0, dtype=float)
            mean_leaked[:, 1] += np.asarray(x_pre_readout_history == 3, dtype=float)

        b_z_fail = False
        if z_bpd is not None:
            z_detector_history, z_unsatisfied_fraction = get_detector_history(
                z_syndrome_history,
                z_syndrome_map,
                x_checks,
                n2,
                n_cycles,
                fnc,
                b_initial_state,
            )
            assert HZ_decoder.shape[0] == len(z_detector_history)
            if relay_decoder == "":
                z_bpd.decode(z_detector_history)
                z_low_weight_error = z_bpd.osdw_decoding
                b_converged = 1 if z_bpd.converge else 0
                bp_iterations[i_shot, 0] = z_bpd.iter
                z_details = None
            else:
                z_details = z_bpd.decode_detailed(
                    np.asarray(z_detector_history, dtype=np.uint8)
                )
                z_low_weight_error = z_details.decoding
                b_converged = 1 if z_details.success else 0
                bp_iterations[i_shot, 0] = z_details.iterations

            bp_converged[i_shot, 0] = b_converged
            if not b_converged:
                bp_non_converged[0][i_shot] = False
            unsatisfied_fraction[i_shot, 0] = z_unsatisfied_fraction
            assert len(z_low_weight_error) == HZ.shape[1]
            z_syndrome_history_augmented_guessed = (HZ @ z_low_weight_error) % 2
            z_syndrome_final_logical_guessed = z_syndrome_history_augmented_guessed[
                z_logical_row : (z_logical_row + k)
            ]
            z_state_data_qubits = [z_state[lin_order[q]] for q in data_qubits]
            z_syndrome_final_logical = (lx @ z_state_data_qubits) % 2
            b_z_fail = not np.array_equal(
                z_syndrome_final_logical_guessed, z_syndrome_final_logical
            )

        b_x_fail = False
        if x_bpd is not None:
            x_detector_history, x_unsatisfied_fraction = get_detector_history(
                x_syndrome_history,
                x_syndrome_map,
                z_checks,
                n2,
                n_cycles,
                fnc,
                b_initial_state,
            )
            assert HX_decoder.shape[0] == len(x_detector_history)
            if relay_decoder == "":
                x_bpd.decode(x_detector_history)
                x_low_weight_error = x_bpd.osdw_decoding
                b_converged = 1 if x_bpd.converge else 0
                bp_iterations[i_shot, 1] = x_bpd.iter
                x_details = None
            else:
                x_detector_history_u8 = np.asarray(x_detector_history, dtype=np.uint8)
                # print("np.sum(x_detector_history_u8): ", np.sum(x_detector_history_u8))
                x_details = x_bpd.decode_detailed(x_detector_history_u8)
                x_low_weight_error = x_details.decoding
                b_converged = 1 if x_details.success else 0
                bp_iterations[i_shot, 1] = x_details.iterations
                # s_output_path, _, s_simulation_path, _ = generate_decoding_paths()
                # s_file_name = s_output_path + f"x_detectors_c_{1}.npy"
                # np.save(s_file_name, x_detector_history_u8)

            bp_converged[i_shot, 1] = b_converged
            if not b_converged:
                bp_non_converged[1][i_shot] = False
            unsatisfied_fraction[i_shot, 1] = x_unsatisfied_fraction
            assert len(x_low_weight_error) == HX.shape[1]
            x_syndrome_history_augmented_guessed = (HX @ x_low_weight_error) % 2
            x_syndrome_final_logical_guessed = x_syndrome_history_augmented_guessed[
                x_logical_row : (x_logical_row + k)
            ]
            x_state_data_qubits = [x_state[lin_order[q]] for q in data_qubits]
            x_syndrome_final_logical = (lz @ x_state_data_qubits) % 2
            b_x_fail = not np.array_equal(
                x_syndrome_final_logical_guessed, x_syndrome_final_logical
            )

        if b_z_fail:
            shot_fails[0][i_shot] = False
            bad_shots_z += 1
        if b_x_fail:
            shot_fails[1][i_shot] = False
            bad_shots_x += 1
        # print("")
        if b_z_fail or b_x_fail:
            bad_shots += 1
            print(f"Shot #{i_shot + 1}, {bad_shots} bad shots.")
        elif (i_shot % output_step) == (output_step - 1):
            print(f"Shot #{i_shot + 1}, {bad_shots} bad shots.")

    mean_ground /= n_shots
    mean_leaked /= n_shots
    t2 = datetime.now()
    duration = (t2 - t1).total_seconds()
    shot_time = duration / n_shots
    cycle_err = bad_shots / (n_shots * n_cycles)
    shots_err = (cycle_err * (1.0 - cycle_err) / (n_shots * n_cycles)) ** 0.5

    print(f"\nReadout flip: {b_readout_flip}.")
    print(f"Number of cycles: {n_cycles}.")
    print(
        f"Simulation duration: {round(duration, 2)}s, " f"{round(shot_time, 4)} s/shot."
    )
    print(
        f"Mean Z, X statistics:\n"
        f"BP iterations: {np.mean(bp_iterations, 0)}.\n"
        f"BP converged fraction: {np.mean(bp_converged, 0)}.\n"
        f"Unsatisfied-checks fraction: {np.mean(unsatisfied_fraction, 0)}.\n"
        f"Pre-readout ground state population: {np.mean(mean_ground, 0)}.\n"
        f"Pre-readout leaked state population: {np.mean(mean_leaked, 0)}.\n"
    )

    results = {
        "n_shots": n_shots,
        "duration": duration,
        "bad_shots": bad_shots,
        "shot_time": shot_time,
        "cycle_err": cycle_err,
        "shots_err": shots_err,
        "bad_shots_z": bad_shots_z,
        "bad_shots_x": bad_shots_x,
        "noise_model": noise_model,
        "mean_ground_z_x": mean_ground,
        "mean_leaked_z_x": mean_leaked,
        "shot_fails_z_x": shot_fails,
        "bp_non_converged_z_x": bp_non_converged,
        "bp_iterations_z_x": bp_iterations,
    }

    if partial_decoding == "":
        converged_mean = np.mean(bp_converged)
        iterations_mean = np.mean(bp_iterations)
    else:
        converged_mean = np.mean(bp_converged[:, 0 if partial_decoding == "z" else 1])
        iterations_mean = np.mean(bp_iterations[:, 0 if partial_decoding == "z" else 1])
    summary = {
        "duration": duration,
        "bad_shots": bad_shots,
        "shot_time": shot_time,
        "bp_converged": converged_mean,
        "bp_iterations": iterations_mean,
        "cycle_err": cycle_err,
        "shots_err": shots_err,
        "mean_ground": np.mean(mean_ground),
        "mean_leaked": np.mean(mean_leaked),
    }
    return results, summary


def create_decoders(simulation_input: Dict, decoder_data: Dict):
    """
    Initialize BP-OSD or Relay BP decoder instances for X and Z sectors.
    
    Args:
        simulation_input: Decoder configuration parameters.
        decoder_data: Decoder data with parity check matrices and error priors.
        
    Returns:
        Tuple of (x_decoder, z_decoder) instances.
    """
    rand_seed = simulation_input.get("rand_seed", 42)
    relay_decoder = simulation_input.get("relay_decoder", "")
    partial_decoding = simulation_input.get("partial_decoding", "").lower()

    # BP-OSD decoder parameters
    bp_method = simulation_input["bp_method"]
    osd_method = simulation_input["osd_method"]  # "osd_e", "osd_cs", "osd0"
    osd_order = simulation_input["osd_order"]  # The osd search depth
    ms_scaling_factor = simulation_input["ms_scaling_factor"]
    # min-sum scaling factor. If 0, a variable-scaling factor method is used
    bp_max_iterations = simulation_input["bp_max_iterations"]  # BP iterations cutoff
    gamma0 = simulation_input.get(
        "gamma0", 0.0
    )  # Uniform memory weight for the first ensemble
    pre_iter = simulation_input.get(
        "pre_iter", 80
    )  # Max BP iterations for the first ensemble
    num_sets = simulation_input.get(
        "num_sets", 301
    )  # Number of relay ensemble elements
    set_max_iter = simulation_input.get(
        "set_max_iter", 60
    )  # Max BP iterations per relay ensemble
    gamma_dist_interval = simulation_input.get(
        "gamma_dist_interval", (-0.24, 0.66)
    )  # range for disordered memory weight selection
    stop_nconv = simulation_input.get(
        "stop_nconv", 5
    )  # Number of relay solutions to find before stopping

    HX_decoder = decoder_data["HX_decoder"]
    HZ_decoder = decoder_data["HZ_decoder"]
    x_probs = decoder_data["x_probs"]
    z_probs = decoder_data["z_probs"]

    x_bpd = None
    z_bpd = None
    if relay_decoder == "":
        if partial_decoding == "" or partial_decoding == "x":
            x_bpd = bposd_decoder(
                HX_decoder,  # the parity check matrix
                channel_probs=x_probs,
                max_iter=int(float(bp_max_iterations)),
                bp_method=bp_method,
                ms_scaling_factor=float(ms_scaling_factor),
                osd_method=osd_method,
                osd_order=int(float(osd_order)),
            )
        if partial_decoding == "" or partial_decoding == "z":
            z_bpd = bposd_decoder(
                HZ_decoder,  # the parity check matrix
                channel_probs=z_probs,
                max_iter=int(float(bp_max_iterations)),
                bp_method=bp_method,
                ms_scaling_factor=float(ms_scaling_factor),
                osd_method="osd_cs",
                osd_order=int(float(osd_order)),
            )
    else:
        if relay_decoder == "RelayDecoderF64":
            relay_class = relay_bp.RelayDecoderF64
        elif relay_decoder == "RelayDecoderF32":
            relay_class = relay_bp.RelayDecoderF32
        elif relay_decoder == "RelayDecoderI64":
            relay_class = relay_bp.RelayDecoderI64
        elif relay_decoder == "MinSumBPDecoderF32":
            relay_class = relay_bp.MinSumBPDecoderF32
        else:
            raise Exception(
                f"The relay_decoder parameter '{relay_decoder}' is unsupported"
            )
        if "BP" in relay_decoder:
            if partial_decoding == "" or partial_decoding == "x":
                x_bpd = relay_class(
                    csr_matrix(HX_decoder),
                    error_priors=np.asarray(
                        x_probs
                    ),  # Set the priors probability for each error
                    gamma0=float(
                        gamma0
                    ),  # Uniform memory weight for the first ensemble
                    max_iter=int(float(bp_max_iterations)),  # Max BP iterations
                )
            if partial_decoding == "" or partial_decoding == "z":
                z_bpd = relay_class(
                    csr_matrix(HZ_decoder),
                    error_priors=np.asarray(
                        z_probs
                    ),  # Set the priors probability for each error
                    gamma0=float(
                        gamma0
                    ),  # Uniform memory weight for the first ensemble
                    max_iter=int(float(bp_max_iterations)),  # Max BP iterations
                )
        else:
            if isinstance(gamma_dist_interval, str):
                gamma_dist_interval = tuple(
                    json.loads(gamma_dist_interval.replace("(", "[").replace(")", "]"))
                )
            if partial_decoding == "" or partial_decoding == "x":
                x_bpd = relay_class(
                    csr_matrix(HX_decoder),
                    error_priors=np.asarray(
                        x_probs
                    ),  # Set the priors probability for each error
                    gamma0=float(
                        gamma0
                    ),  # Uniform memory weight for the first ensemble
                    pre_iter=int(
                        float(pre_iter)
                    ),  # Max BP iterations for the first ensemble
                    num_sets=int(float(num_sets)),  # Number of relay ensemble elements
                    set_max_iter=int(
                        float(set_max_iter)
                    ),  # Max BP iterations per relay ensemble
                    gamma_dist_interval=gamma_dist_interval,
                    # Set the uniform distribution range for disordered memory weight selection
                    stop_nconv=int(
                        float(stop_nconv)
                    ),  # Number of relay solutions to find before stopping
                    seed=rand_seed,
                )
            if partial_decoding == "" or partial_decoding == "z":
                z_bpd = relay_class(
                    csr_matrix(HZ_decoder),
                    error_priors=np.asarray(
                        z_probs
                    ),  # Set the priors probability for each error
                    gamma0=float(
                        gamma0
                    ),  # Uniform memory weight for the first ensemble
                    pre_iter=int(
                        float(pre_iter)
                    ),  # Max BP iterations for the first ensemble
                    num_sets=int(float(num_sets)),  # Number of relay ensemble elements
                    set_max_iter=int(
                        float(set_max_iter)
                    ),  # Max BP iterations per relay ensemble
                    gamma_dist_interval=gamma_dist_interval,
                    # Set the uniform distribution range for disordered memory weight selection
                    stop_nconv=int(
                        float(stop_nconv)
                    ),  # Number of relay solutions to find before stopping
                    seed=rand_seed,
                )
    return x_bpd, z_bpd


def simulate_detector_faults(
    simulation_input: Dict,
    decoder_data: Dict,
    z_fault_column_lists: Optional[List[List]] = None,
    x_fault_column_lists: Optional[List[List]] = None,
) -> (Dict, Dict):
    """
    Test decoder on specific fault patterns without circuit simulation.
    
    Directly constructs detector syndromes from fault column combinations and
    attempts decoding. Used to identify decoder escape patterns.
    
    Args:
        simulation_input: Decoder configuration.
        decoder_data: Decoder data structure.
        z_fault_column_lists: List of Z-fault column combinations to test.
        x_fault_column_lists: List of X-fault column combinations to test.
        
    Returns:
        Tuple of (results_dict, summary_dict) with decoding statistics.
    """
    output_step = simulation_input.get("output_step", 50)
    # Number of shots to skip between console output lines if no error occurred
    weight = simulation_input.get("weight", "")
    relay_decoder = simulation_input.get("relay_decoder", "")
    partial_decoding = simulation_input.get("partial_decoding", "").lower()
    n_z_faults = len(z_fault_column_lists) if z_fault_column_lists is not None else 0
    n_x_faults = len(x_fault_column_lists) if x_fault_column_lists is not None else 0
    n_shots = max(n_x_faults, n_z_faults)

    HX_decoder = decoder_data["HX_decoder"]
    HZ_decoder = decoder_data["HZ_decoder"]
    HX = decoder_data["HX"]
    HZ = decoder_data["HZ"]
    z_logical_row = decoder_data["z_logical_row"]
    x_logical_row = decoder_data["x_logical_row"]
    k = decoder_data["k"]
    n_cycles = decoder_data["n_cycles"]

    x_bpd, z_bpd = create_decoders(simulation_input, decoder_data)
    bad_shots = 0
    bad_shots_z = 0
    bad_shots_x = 0
    t1 = datetime.now()
    shot_fails = [{}, {}]
    bp_non_converged = [{}, {}]
    bp_converged = np.full((n_shots, 2), 0, dtype=int)
    bp_iterations = np.full((n_shots, 2), 0, dtype=int)
    b_z_decoding = z_bpd is not None and z_fault_column_lists is not None
    b_x_decoding = x_bpd is not None and x_fault_column_lists is not None
    z_detector_histories = (
        np.zeros((n_shots, HZ_decoder.shape[0])) if b_z_decoding else None
    )
    x_detector_histories = (
        np.zeros((n_shots, HX_decoder.shape[0])) if b_x_decoding else None
    )
    print(
        f"Running fault cols decoding search for, with weight = {weight}, {n_shots} faults."
    )
    for i_shot in range(n_shots):
        b_z_fail = False
        if b_z_decoding:
            z_detector_history = 0 * HZ_decoder.getcol(0).toarray().flatten()
            z_syndrome_final_logical = 0 * HZ.getcol(0).toarray().flatten()
            fault_cols = z_fault_column_lists[i_shot]
            for fault_col in fault_cols:
                decoder_col = HZ_decoder.getcol(fault_col).toarray().flatten()
                z_detector_history = (z_detector_history + decoder_col) % 2
                hz_col = HZ.getcol(fault_col).toarray().flatten()
                z_syndrome_final_logical = (z_syndrome_final_logical + hz_col) % 2
            z_syndrome_final_logical = z_syndrome_final_logical[
                z_logical_row : (z_logical_row + k)
            ]
            z_detector_histories[i_shot, :] = z_detector_history
            if relay_decoder == "":
                z_bpd.decode(z_detector_history)
                z_low_weight_error = z_bpd.osdw_decoding
                b_converged = 1 if z_bpd.converge else 0
                bp_iterations[i_shot, 0] = z_bpd.iter
            else:
                z_details = z_bpd.decode_detailed(
                    np.asarray(z_detector_history, dtype=np.uint8)
                )
                z_low_weight_error = z_details.decoding
                b_converged = 1 if z_details.success else 0
                bp_iterations[i_shot, 0] = z_details.iterations

            bp_converged[i_shot, 0] = b_converged
            if not b_converged:
                bp_non_converged[0][i_shot] = False
            assert len(z_low_weight_error) == HZ.shape[1]
            z_syndrome_history_augmented_guessed = (HZ @ z_low_weight_error) % 2
            z_syndrome_final_logical_guessed = z_syndrome_history_augmented_guessed[
                z_logical_row : (z_logical_row + k)
            ]
            b_z_fail = not np.array_equal(
                z_syndrome_final_logical_guessed, z_syndrome_final_logical
            )

        b_x_fail = False
        if b_x_decoding:
            x_detector_history = 0 * HX_decoder.getcol(0).toarray().flatten()
            x_syndrome_final_logical = 0 * HX.getcol(0).toarray().flatten()
            fault_cols = x_fault_column_lists[i_shot]
            for fault_col in fault_cols:
                decoder_col = HX_decoder.getcol(fault_col).toarray().flatten()
                x_detector_history = (x_detector_history + decoder_col) % 2
                hx_col = HX.getcol(fault_col).toarray().flatten()
                x_syndrome_final_logical = (x_syndrome_final_logical + hx_col) % 2
            x_syndrome_final_logical = x_syndrome_final_logical[
                x_logical_row : (x_logical_row + k)
            ]
            if relay_decoder == "":
                x_bpd.decode(x_detector_history)
                x_low_weight_error = x_bpd.osdw_decoding
                b_converged = 1 if x_bpd.converge else 0
                bp_iterations[i_shot, 1] = x_bpd.iter
            else:
                x_detector_history_u8 = np.asarray(x_detector_history, dtype=np.uint8)
                x_details = x_bpd.decode_detailed(x_detector_history_u8)
                x_low_weight_error = x_details.decoding
                b_converged = 1 if x_details.success else 0
                bp_iterations[i_shot, 1] = x_details.iterations
            x_detector_histories[i_shot, :] = x_detector_history

            bp_converged[i_shot, 1] = b_converged
            if not b_converged:
                bp_non_converged[1][i_shot] = False
            assert len(x_low_weight_error) == HX.shape[1]
            x_syndrome_history_augmented_guessed = (HX @ x_low_weight_error) % 2
            x_syndrome_final_logical_guessed = x_syndrome_history_augmented_guessed[
                x_logical_row : (x_logical_row + k)
            ]
            b_x_fail = not np.array_equal(
                x_syndrome_final_logical_guessed, x_syndrome_final_logical
            )

        if b_z_fail:
            shot_fails[0][i_shot] = False
            bad_shots_z += 1
            print(z_fault_column_lists[i_shot])
        if b_x_fail:
            shot_fails[1][i_shot] = False
            bad_shots_x += 1
            print(x_fault_column_lists[i_shot])
        if b_z_fail or b_x_fail:
            bad_shots += 1
            print(f"Shot #{i_shot + 1}, {bad_shots} bad shots.")
        elif (i_shot % output_step) == (output_step - 1):
            print(f"Shot #{i_shot + 1}, {bad_shots} bad shots.")

    t2 = datetime.now()
    duration = (t2 - t1).total_seconds()
    shot_time = duration / n_shots
    cycle_err = bad_shots / (n_shots * n_cycles)
    shots_err = (cycle_err * (1.0 - cycle_err) / (n_shots * n_cycles)) ** 0.5

    print(f"Number of cycles: {n_cycles}.")
    print(
        f"Simulation duration: {round(duration, 2)}s, " f"{round(shot_time, 4)} s/shot."
    )
    print(
        f"Mean Z, X statistics:\n"
        f"BP iterations: {np.mean(bp_iterations, 0)}.\n"
        f"BP converged fraction: {np.mean(bp_converged, 0)}.\n"
    )

    results = {
        "n_shots": n_shots,
        "duration": duration,
        "bad_shots": bad_shots,
        "shot_time": shot_time,
        "cycle_err": cycle_err,
        "shots_err": shots_err,
        "bad_shots_z": bad_shots_z,
        "bad_shots_x": bad_shots_x,
        "shot_fails_z_x": shot_fails,
        "bp_non_converged_z_x": bp_non_converged,
        "bp_iterations_z_x": bp_iterations,
        "fault_column_lists_z": z_fault_column_lists,
        "fault_column_lists_x": x_fault_column_lists,
        "detector_histories_z": z_detector_histories,
        "detector_histories_x": x_detector_histories,
    }

    if partial_decoding == "":
        converged_mean = np.mean(bp_converged)
        iterations_mean = np.mean(bp_iterations)
    else:
        converged_mean = np.mean(bp_converged[:, 0 if partial_decoding == "z" else 1])
        iterations_mean = np.mean(bp_iterations[:, 0 if partial_decoding == "z" else 1])
    summary = {
        "duration": duration,
        "bad_shots": bad_shots,
        "shot_time": shot_time,
        "bp_converged": converged_mean,
        "bp_iterations": iterations_mean,
        "n_shots": n_shots,
    }
    return results, summary


def get_pair_cancelled_cols(check_pair, i_0, check_cols_union, H_decoder):
    """
    Find fault columns that cancel (XOR to zero) for a pair of detector checks.
    
    Args:
        check_pair: Tuple of two check indices.
        i_0: Offset for check indexing.
        check_cols_union: Dictionary mapping check pairs to column unions.
        H_decoder: Parity check matrix.
        
    Returns:
        List of column indices that trigger both checks in the pair.
    """
    i = check_pair[0]
    j = check_pair[1]
    cols_union = check_cols_union[(i_0 + i, i_0 + j)]
    cancelled_cols = []
    for col_i in cols_union:
        col = H_decoder.getcol(col_i).toarray().flatten()
        if col[i_0 + i] and col[i_0 + j]:
            if col_i not in cancelled_cols:
                cancelled_cols.append(col_i)
    return cancelled_cols


def generate_pair_shared_cols_matrix(
    H_decoder, rows, filter_pairs_by_shared_cols_count: Optional[List[int]] = None
):
    """
    Generate matrix of shared fault columns between all pairs of detector checks.
    
    Identifies which fault columns trigger multiple detectors simultaneously,
    useful for finding low-weight error patterns that may escape decoding.
    
    Args:
        H_decoder: Parity check matrix.
        rows: List of detector row indices to analyze.
        filter_pairs_by_shared_cols_count: Optional filter for pairs by shared column count.
        
    Returns:
        Tuple of (shared_cols_dict, count_dict, count_matrix, filtered_pairs).
    """
    n_rows = len(rows)
    pairs_filtered_by_shared_cols_count = []
    pair_shared_cols_dict = {}
    pair_shared_cols_count_dict = {}
    pair_shared_cols_count_matrix = np.full((n_rows, n_rows), np.nan)
    for i, c_1 in enumerate(rows):  # TODO: Need to generalize to nonconsecutive cols
        for j, c_2 in enumerate(rows):
            if j <= i:
                continue
            faults_i = list(H_decoder.getrow(c_1).indices)
            faults_j = list(H_decoder.getrow(c_2).indices)
            faults = set(faults_i + faults_j)
            len_i = len(faults_i)
            len_j = len(faults_j)
            len_faults = len(faults)
            pair_cancelled_cols = []
            if len_faults < len_i + len_j:
                for col_i in faults:
                    col = H_decoder.getcol(col_i)
                    if (
                        col[c_1, 0] and col[c_2, 0]
                    ):  # Could be made efficient with outer loop
                        if col_i not in pair_cancelled_cols:
                            pair_cancelled_cols.append(col_i)
            pair_shared_cols_dict[(c_1, c_2)] = pair_cancelled_cols
            pair_shared_cols_dict[(c_2, c_1)] = pair_cancelled_cols
            n_len = len(pair_cancelled_cols)
            pair_shared_cols_count_dict[(c_1, c_2)] = n_len
            pair_shared_cols_count_dict[(c_2, c_1)] = n_len
            pair_shared_cols_count_matrix[i, j] = n_len
            pair_shared_cols_count_matrix[j, i] = n_len
            if (
                filter_pairs_by_shared_cols_count is None
                or pair_shared_cols_count_matrix[i, j]
                in filter_pairs_by_shared_cols_count
            ):
                pairs_filtered_by_shared_cols_count.append((c_1, c_2))
    return (
        pair_shared_cols_dict,
        pair_shared_cols_count_dict,
        pair_shared_cols_count_matrix,
        pairs_filtered_by_shared_cols_count,
    )


def add_fault_columns(H_decoder, fault_cols, fault_syndrome=None, fault_checks=None):
    """
    Compute the combined syndrome from multiple fault columns.
    
    Args:
        H_decoder: Parity check matrix.
        fault_cols: List of fault column indices to combine.
        fault_syndrome: Optional initial syndrome to add to.
        fault_checks: Optional list to accumulate triggered checks.
        
    Returns:
        Tuple of (n_canceled, n_original, syndrome, checks, weights).
    """
    if fault_checks is None:
        fault_checks = []
    if fault_syndrome is None:
        fault_syndrome = 0 * H_decoder.getcol(0).toarray().flatten()
    original_weights = []
    for fault_col in fault_cols:
        decoder_col = H_decoder.getcol(fault_col).toarray().flatten()
        original_weights.append(decoder_col.sum())
        fault_syndrome = (fault_syndrome + decoder_col) % 2
        fault_checks.extend(np.where(decoder_col)[0])
    syndrome_weight = np.sum(fault_syndrome)
    unique_checks = set(
        fault_checks
    )  # Do not double count same check appearing in multiple columns
    n_original_checks = len(unique_checks)
    n_canceled_checks = n_original_checks - syndrome_weight
    return (
        n_canceled_checks,
        n_original_checks,
        fault_syndrome,
        fault_checks,
        original_weights,
    )


def choose_detector_faults_weight_4(
    decoder_data,
    s_logical,
    rows=None,
    n_fault_repeat=1,
    pair_0_list: Optional[Union[int, List[Tuple[int, int]]]] = None,
    filter_pairs_by_shared_cols_count: Optional[List[int]] = None,
    filter_syndromes_by_total_canceled_count: Optional[List[int]] = None,
    filter_syndromes_by_pair_canceled_counts: Optional[List[List[int]]] = None,
):
    """
    Generate weight-4 fault patterns for decoder testing.
    
    Systematically constructs combinations of 4 fault columns from pairs of
    detector checks, filtering by cancellation patterns. Used to find low-weight
    errors that may escape the decoder.
    
    Args:
        decoder_data: Decoder data structure.
        s_logical: "X" or "Z" to specify error type.
        rows: Detector rows to analyze (default: checks relevant for first cycle).
        n_fault_repeat: Number of times to repeat each fault pattern.
        pair_0_list: Initial check pairs to start from.
        filter_pairs_by_shared_cols_count: Filter pairs by shared column count.
        filter_syndromes_by_total_canceled_count: Filter by total cancellations.
        filter_syndromes_by_pair_canceled_counts: Filter by per-pair cancellations.
        
    Returns:
        Tuple of arrays with fault patterns and their properties.
    """
    s_decoder_type = s_logical[0].lower()
    H_decoder = decoder_data["HX_decoder" if s_decoder_type == "x" else "HZ_decoder"]
    if rows is None:
        n_rows = decoder_data["n"]
        if s_decoder_type == "z":
            rows = range(0, int(n_rows / 2))
        else:
            rows = range(0, n_rows)

    (
        pair_shared_cols_dict,
        pair_shared_cols_count_dict,
        pair_shared_cols_count_matrix,
        pairs_filtered_by_shared_cols_count,
    ) = generate_pair_shared_cols_matrix(
        H_decoder, rows, filter_pairs_by_shared_cols_count
    )
    if pair_0_list is None:
        pair_0_list = [pairs_filtered_by_shared_cols_count[0]]
    elif type(pair_0_list) is int:
        pair_0_list = pairs_filtered_by_shared_cols_count[0:pair_0_list]
    r_repeat = range(n_fault_repeat)
    faults = []
    faults_pairs = []
    faults_to_pair_1 = []
    faults_original_weights = []
    faults_weight = []
    faults_original_pair_checks_count = []
    faults_canceled_pair_checks_count = []
    faults_canceled_total_checks_count = []
    for pair_0 in pair_0_list:
        cols_to_choose_0 = pair_shared_cols_dict[pair_0[0], pair_0[1]]
        faults_0 = []
        for comb in itertools.combinations(cols_to_choose_0, 2):
            faults_0.append(list(comb))

        for fault_0 in faults_0:
            fault_0_set = set(fault_0)
            (
                n_canceled_checks_0,
                n_original_checks_0,
                fault_syndrome_0,
                fault_checks_0,
                original_weights_0,
            ) = add_fault_columns(H_decoder, fault_0)
            for i_pair_1, pair_1 in enumerate(pairs_filtered_by_shared_cols_count):
                if pair_1 == pair_0:
                    continue
                cols_to_choose_1 = pair_shared_cols_dict[pair_1[0], pair_1[1]]
                for comb_1 in itertools.combinations(cols_to_choose_1, 2):
                    fault_1 = list(comb_1)
                    fault_1_set = set(fault_1)
                    fault = fault_0_set | fault_1_set
                    if len(fault) < 4:
                        continue
                    (
                        n_canceled_checks_1,
                        n_original_checks_1,
                        fault_syndrome_1,
                        fault_checks_1,
                        original_weights_1,
                    ) = add_fault_columns(H_decoder, fault_1)
                    fault_checks = fault_checks_0 + fault_checks_1
                    unique_checks = set(fault_checks)
                    fault_syndrome = (fault_syndrome_0 + fault_syndrome_1) % 2
                    fault_weight = np.sum(fault_syndrome)
                    n_original_checks = len(unique_checks)
                    n_canceled_checks = n_original_checks - fault_weight
                    original_weights = original_weights_0 + original_weights_1
                    if (
                        filter_syndromes_by_total_canceled_count is None
                        or n_canceled_checks in filter_syndromes_by_total_canceled_count
                    ):
                        original_checks_count = [
                            n_original_checks_0,
                            n_original_checks_1,
                        ]
                        canceled_checks_count = [
                            n_canceled_checks_0,
                            n_canceled_checks_1,
                        ]
                        if fault not in faults and (
                            filter_syndromes_by_pair_canceled_counts is None
                            or canceled_checks_count
                            in filter_syndromes_by_pair_canceled_counts
                        ):
                            if fault_0_set not in faults_pairs:
                                faults_pairs.append(fault_0_set)
                            if fault_1_set not in faults_pairs:
                                faults_pairs.append(fault_1_set)
                            for _ in r_repeat:
                                faults.append(fault)
                                faults_to_pair_1.append(i_pair_1)
                                faults_canceled_total_checks_count.append(
                                    n_canceled_checks
                                )
                                faults_original_pair_checks_count.append(
                                    original_checks_count
                                )
                                faults_canceled_pair_checks_count.append(
                                    canceled_checks_count
                                )
                                faults_original_weights.append(original_weights)
                                faults_weight.append(fault_weight)

    return (
        np.asarray(faults),
        np.asarray(faults_pairs),
        np.asarray(faults_to_pair_1),
        pair_shared_cols_count_matrix,
        np.asarray(faults_canceled_total_checks_count),
        np.asarray(faults_original_pair_checks_count),
        np.asarray(faults_canceled_pair_checks_count),
        np.asarray(faults_original_weights),
        np.asarray(faults_weight),
    )
