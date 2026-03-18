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
Decoder data setup for Bivariate Bicycle codes.

This module constructs the decoder data structures for BB codes, including:
- Code definition from polynomial parameters
- Logical-idle syndrome measurement circuit generation
- Parity check matrix construction for decoding
"""

import numpy as np
from typing import Dict
from bposd.css import css_code
from scipy.sparse import coo_matrix
from scipy.sparse import hstack

from noise_model import NoiseModel
from circuit_simulation import (
    generate_linearized_faulty_circuits,
    simulate_x_errors,
    simulate_z_errors,
)


def rank2(A):
    """
    Compute the rank of a binary matrix over GF(2).

    Args:
        A: Binary matrix (numpy array).

    Returns:
        Rank of A over the binary field F_2.
    """
    rows, n = A.shape
    X = np.identity(n, dtype=int)

    for i in range(rows):
        y = np.dot(A[i, :], X) % 2
        not_y = (y + 1) % 2
        good = X[:, np.nonzero(not_y)]
        good = good[:, 0, :]
        bad = X[:, np.nonzero(y)]
        bad = bad[:, 0, :]
        if bad.shape[1] > 0:
            bad = np.add(bad, np.roll(bad, 1, axis=1))
            bad = bad % 2
            bad = np.delete(bad, 0, axis=1)
            X = np.concatenate((good, bad), axis=1)
    # now columns of X span the binary null-space of A
    return n - X.shape[1]


def generate_decoder_data(decoder_input: Dict, noise_model: NoiseModel) -> Dict:
    """
    Generates a complete decoder data dictionary for the repeated idle syndrome cycle.

    Args:
        decoder_input: Dictionary with keys:
            - "code_name": Code identifier (e.g., "144.12.12" for n=144, k=12, d=12)
            - "n_cycles": Number of syndrome measurement cycles
            - "fnc": Number of final noiseless cycles for validation
            - "unique_id": UUID for this decoder configuration
        noise_model: Noise model specifying error rates for all gates.

    Returns:
        Complete decoder data dictionary containing code parameters, circuit,
        parity check matrices (HX_decoder, HZ_decoder), and fault mappings.
    """

    # Parameters of a Bivariate Bicycle (BB) code
    # see Section 4 of https://arxiv.org/pdf/2308.07915.pdf for notations
    # The code is defined by a pair of polynomials
    # A and B that depends on two variables x and y such that
    # x^ell = 1
    # y^m = 1
    # A = x^{a_1} + y^{a_2} + y^{a_3}
    # B = y^{b_1} + x^{b_2} + x^{b_3}

    b_amend_decoder = False

    s_code_name: str = decoder_input["code_name"]
    n_cycles: int = decoder_input["n_cycles"]
    fnc: int = decoder_input["fnc"]
    rand_seed = decoder_input.get("rand_seed", 42)

    if s_code_name == "72.12.6":
        ell, m = 6, 6
        a1, a2, a3 = 3, 1, 2
        b1, b2, b3 = 3, 1, 2
        d = 6
    elif s_code_name == "144.12.12":
        ell, m = 12, 6
        a1, a2, a3 = 3, 1, 2
        b1, b2, b3 = 3, 1, 2
        d = 12
    elif s_code_name == "288.12.18":
        ell, m = 12, 12
        a1, a2, a3 = 3, 2, 7
        b1, b2, b3 = 3, 1, 2
        d = 18
    elif s_code_name == "784.24.24":
        ell, m = 28, 14
        a1, a2, a3 = 26, 6, 8
        b1, b2, b3 = 7, 9, 20
        d = 24
    else:
        raise Exception("Unsupported code.")
    # code length
    n = 2 * m * ell
    n2 = m * ell

    # Compute check matrices of X- and Z-checks

    # cyclic shift matrices
    I_ell = np.identity(ell, dtype=int)
    I_m = np.identity(m, dtype=int)
    I = np.identity(ell * m, dtype=int)
    x = {}
    y = {}
    for i in range(ell):
        x[i] = np.kron(np.roll(I_ell, i, axis=1), I_m)
    for i in range(m):
        y[i] = np.kron(I_ell, np.roll(I_m, i, axis=1))

    A = (x[a1] + y[a2] + y[a3]) % 2
    B = (y[b1] + x[b2] + x[b3]) % 2

    A1 = x[a1]
    A2 = y[a2]
    A3 = y[a3]
    B1 = y[b1]
    B2 = x[b2]
    B3 = x[b3]

    AT = np.transpose(A)
    BT = np.transpose(B)

    hx = np.hstack((A, B))
    hz = np.hstack((BT, AT))

    # number of logical qubits
    k = n - rank2(hx) - rank2(hz)

    qcode = css_code(hx, hz, d)
    print("Testing CSS code...")
    qcode.test()
    print("Done")

    lz = qcode.lz
    lx = qcode.lx

    # Give a name to each qubit
    # Define a linear order on the set of qubits
    lin_order = {}
    data_qubits = []
    x_checks = []
    z_checks = []
    data_qubit_indices = []
    cnt = 0
    for i in range(n2):
        node_name = ("XC", i)
        x_checks.append(node_name)
        lin_order[node_name] = cnt
        cnt += 1
    for i in range(n2):
        node_name = ("DL", i)
        data_qubits.append(node_name)
        lin_order[node_name] = cnt
        data_qubit_indices.append(cnt)
        cnt += 1
    for i in range(n2):
        node_name = ("DR", i)
        data_qubits.append(node_name)
        lin_order[node_name] = cnt
        data_qubit_indices.append(cnt)
        cnt += 1
    for i in range(n2):
        node_name = ("ZC", i)
        z_checks.append(node_name)
        lin_order[node_name] = cnt
        cnt += 1

    # compute the list of neighbors of each check qubit in the Tanner graph
    nbs = {}
    # iterate over X checks
    for i in range(n2):
        check_name = ("XC", i)
        # left data qubits
        nbs[(check_name, 0)] = ("DL", np.nonzero(A1[i, :])[0][0])
        nbs[(check_name, 1)] = ("DL", np.nonzero(A2[i, :])[0][0])
        nbs[(check_name, 2)] = ("DL", np.nonzero(A3[i, :])[0][0])
        # right data qubits
        nbs[(check_name, 3)] = ("DR", np.nonzero(B1[i, :])[0][0])
        nbs[(check_name, 4)] = ("DR", np.nonzero(B2[i, :])[0][0])
        nbs[(check_name, 5)] = ("DR", np.nonzero(B3[i, :])[0][0])

    # iterate over Z checks
    for i in range(n2):
        check_name = ("ZC", i)
        # left data qubits
        nbs[(check_name, 0)] = ("DL", np.nonzero(B1[:, i])[0][0])
        nbs[(check_name, 1)] = ("DL", np.nonzero(B2[:, i])[0][0])
        nbs[(check_name, 2)] = ("DL", np.nonzero(B3[:, i])[0][0])
        # right data qubits
        nbs[(check_name, 3)] = ("DR", np.nonzero(A1[:, i])[0][0])
        nbs[(check_name, 4)] = ("DR", np.nonzero(A2[:, i])[0][0])
        nbs[(check_name, 5)] = ("DR", np.nonzero(A3[:, i])[0][0])

    # syndrome cycle with 7 CNOT rounds
    # sX and sZ define the order in which X-check and Z-check qubit
    # is coupled with the neighboring data qubits
    # We label the six neighbors of each check qubit in the Tanner graph
    # by integers 0,1,...,5
    x_check_coupling = [-1, 1, 4, 3, 5, 0, 2]
    z_check_coupling = [3, 5, 0, 1, 2, 4, -1]

    cycle = []
    U = np.identity(2 * n, dtype=int)
    # round 0: prep xchecks, CNOT zchecks and data
    t = 0
    for q in x_checks:
        cycle.append(("PX", q))
    data_qubits_cnoted_in_this_round = []
    assert not (z_check_coupling[t] == -1)
    for target in z_checks:
        direction = z_check_coupling[t]
        control = nbs[(target, direction)]
        U[lin_order[target], :] = (
            U[lin_order[target], :] + U[lin_order[control], :]
        ) % 2
        data_qubits_cnoted_in_this_round.append(control)
        cycle.append(("CX", control, target))
    for q in data_qubits:
        if not (q in data_qubits_cnoted_in_this_round):
            cycle.append(("ID_S", q))  # Short idle gate (parallel to CNOT)

    # round 1-5: CNOT xchecks and data, CNOT zchecks and data
    for t in range(1, 6):
        assert not (x_check_coupling[t] == -1)
        for control in x_checks:
            direction = x_check_coupling[t]
            target = nbs[(control, direction)]
            U[lin_order[target], :] = (
                U[lin_order[target], :] + U[lin_order[control], :]
            ) % 2
            cycle.append(("CX", control, target))
        assert not (z_check_coupling[t] == -1)
        for target in z_checks:
            direction = z_check_coupling[t]
            control = nbs[(target, direction)]
            U[lin_order[target], :] = (
                U[lin_order[target], :] + U[lin_order[control], :]
            ) % 2
            cycle.append(("CX", control, target))

    # round 6: CNOT xchecks and data, measure Z checks
    t = 6
    for q in z_checks:
        cycle.append(("MZ", q))
    assert not (x_check_coupling[t] == -1)
    data_qubits_cnoted_in_this_round = []
    for control in x_checks:
        direction = x_check_coupling[t]
        target = nbs[(control, direction)]
        U[lin_order[target], :] = (
            U[lin_order[target], :] + U[lin_order[control], :]
        ) % 2
        cycle.append(("CX", control, target))
        data_qubits_cnoted_in_this_round.append(target)
    for q in data_qubits:
        if not (q in data_qubits_cnoted_in_this_round):
            cycle.append(("ID_L", q))  # Long idle gate (parallel to readout)

    # round 7: all data qubits are idle, Prep Z checks, Meas X checks
    for q in data_qubits:
        cycle.append(("ID_L", q))  # Long idle gate (parallel to readout)
    for q in x_checks:
        cycle.append(("MX", q))
    for q in z_checks:
        cycle.append(("PZ", q))

    # full syndrome measurement circuit
    cycle.append(("END", 0))
    cycle_repeated = n_cycles * cycle

    # test the syndrome measurement circuit

    # implement syndrome measurements using the sequential depth-12 circuit
    V = np.identity(2 * n, dtype=int)
    # first measure all X checks
    for t in range(7):
        if not (x_check_coupling[t] == -1):
            for control in x_checks:
                direction = x_check_coupling[t]
                target = nbs[(control, direction)]
                V[lin_order[target], :] = (
                    V[lin_order[target], :] + V[lin_order[control], :]
                ) % 2
    # next measure all Z checks
    for t in range(7):
        if not (z_check_coupling[t] == -1):
            for target in z_checks:
                direction = z_check_coupling[t]
                control = nbs[(target, direction)]
                V[lin_order[target], :] = (
                    V[lin_order[target], :] + V[lin_order[control], :]
                ) % 2

    if np.array_equal(U, V):
        print("Circuit test: OK.")
    else:
        print("Circuit test: FAIL!")
        exit()

    decoder_data = decoder_input.copy()
    decoder_data.update(
        {
            "n": n,
            "k": k,
            "lin_order": lin_order,
            "data_qubits": data_qubits,
            "x_checks": x_checks,
            "z_checks": z_checks,
            "nbs": nbs,
            "n_neighbors": 6,
        }
    )
    if noise_model.distributions is not None:
        rng = np.random.default_rng(rand_seed)
        noise_model.local_error_rates = NoiseModel.gate_errors_from_distributions(
            rng,
            decoder_data,
            noise_model.distributions,
        )
        print("Gate error rates in the noise model initialized from distributions.")
    # Compute decoding matrices
    (
        z_circuits,
        z_fault_probs,
        z_faults,
        x_circuits,
        x_fault_probs,
        x_faults,
    ) = generate_linearized_faulty_circuits(cycle_repeated, noise_model, n2, False)

    # execute each noisy circuit and compute the syndrome
    # we add two noiseless syndrome cycles at the end
    print("Computing syndrome histories for single-Z-type-fault circuits...")
    cnt = 0
    HZdict = {}
    for i_circ, circ in enumerate(z_circuits):
        full_circuit = circ.copy()
        for i in range(fnc):
            full_circuit += cycle
        syndrome_history, state, syndrome_map = simulate_z_errors(
            full_circuit, n2, lin_order
        )
        assert len(syndrome_history) == n2 * (n_cycles + fnc)
        state_data_qubits = [state[lin_order[q]] for q in data_qubits]
        syndrome_final_logical = (lx @ state_data_qubits) % 2
        # apply syndrome sparsification map
        syndrome_history_copy = syndrome_history.copy()
        for c in x_checks:
            pos = syndrome_map[c]
            assert len(pos) == (n_cycles + fnc)
            for row in range(1, n_cycles + fnc):
                syndrome_history[pos[row]] += syndrome_history_copy[pos[row - 1]]
        syndrome_history %= 2
        syndrome_history_augmented = np.hstack(
            [syndrome_history, syndrome_final_logical]
        )
        supp = tuple(np.nonzero(syndrome_history_augmented)[0])
        if supp in HZdict:
            HZdict[supp].append(cnt)
        else:
            HZdict[supp] = [cnt]
        cnt += 1
    z_logical_row = n2 * (n_cycles + fnc)
    print("Done.")

    # if a subset of columns of HZ are equal, retain only one of these columns
    print("Computing effective noise model for the Z-type-faults decoder...")
    print("Number of distinct Z-syndrome histories: ", len(HZdict))
    HZ = []
    HZ_decoder = []
    z_probs = []
    z_faults_map = {}
    for supp in HZdict:
        new_column = np.zeros((n2 * (n_cycles + fnc) + k, 1), dtype=int)
        new_column_short = np.zeros((n2 * (n_cycles + fnc), 1), dtype=int)
        new_column[list(supp), 0] = 1
        new_column_short[:, 0] = new_column[0:z_logical_row, 0]
        for i in HZdict[supp]:
            z_faults_map[z_faults[i]] = len(HZ_decoder)
        HZ.append(coo_matrix(new_column))
        HZ_decoder.append(coo_matrix(new_column_short))
        z_probs.append(np.sum([z_fault_probs[i] for i in HZdict[supp]]))
    print("Done.")
    HZ = hstack(HZ)
    HZ_decoder = hstack(HZ_decoder)
    print("Decoding matrix HZ sparseness:")
    print("max col weight=", np.max(np.sum(HZ_decoder, 0)))
    print("max row weight=", np.max(np.sum(HZ_decoder, 1)))

    # execute each noisy circuit and compute the syndrome
    # we add two noiseless syndrome cycles at the end
    print("Computing syndrome histories for single-X-type-fault circuits...")
    cnt = 0
    HXdict = {}
    # syn_supp = []
    for i_circ, circ in enumerate(x_circuits):
        full_circuit = circ.copy()
        for i in range(fnc):
            full_circuit += cycle
        syndrome_history, state, syndrome_map = simulate_x_errors(
            full_circuit, n2, lin_order
        )
        assert len(syndrome_history) == n2 * (n_cycles + fnc)
        state_data_qubits = [state[lin_order[q]] for q in data_qubits]
        syndrome_final_logical = (lz @ state_data_qubits) % 2
        # apply syndrome sparsification map
        syndrome_history_copy = syndrome_history.copy()
        for c in z_checks:
            pos = syndrome_map[c]
            assert len(pos) == (n_cycles + fnc)
            for row in range(1, n_cycles + fnc):
                syndrome_history[pos[row]] += syndrome_history_copy[pos[row - 1]]
        syndrome_history %= 2
        syndrome_history_augmented = np.hstack(
            [syndrome_history, syndrome_final_logical]
        )
        supp = tuple(np.nonzero(syndrome_history_augmented)[0])
        if supp in HXdict:
            HXdict[supp].append(cnt)
        else:
            HXdict[supp] = [cnt]
        # if i_circ in [1647, 1656, 1665, 1683]:
        #     syn_supp.append(supp)
        cnt += 1
    x_logical_row = n2 * (n_cycles + fnc)
    print("Done.")

    # if a subset of columns of H are equal, retain only one of these columns
    print("Computing effective noise model for the X-type-faults decoder...")
    print("Number of distinct X-syndrome histories: ", len(HXdict))
    HX = []
    HX_decoder = []
    x_probs = []
    x_faults_map = {}
    for supp in HXdict:
        new_column = np.zeros((n2 * (n_cycles + fnc) + k, 1), dtype=int)
        new_column_short = np.zeros((n2 * (n_cycles + fnc), 1), dtype=int)
        new_column[list(supp), 0] = 1
        new_column_short[:, 0] = new_column[0:x_logical_row, 0]
        for i in HXdict[supp]:
            x_faults_map[x_faults[i]] = len(HX_decoder)
        HX.append(coo_matrix(new_column))
        HX_decoder.append(coo_matrix(new_column_short))
        x_probs.append(np.sum([x_fault_probs[i] for i in HXdict[supp]]))
    print("Done.")
    HX = hstack(HX)
    HX_decoder = hstack(HX_decoder)
    print("Decoding matrix HX sparseness:")
    print("max col weight: ", np.max(np.sum(HX_decoder, 0)))
    print("max row weight: ", np.max(np.sum(HX_decoder, 1)))

    # save decoding matrices
    decoder_data["data_qubit_indices"] = data_qubit_indices
    decoder_data["HX_decoder"] = HX_decoder
    decoder_data["HZ_decoder"] = HZ_decoder
    decoder_data["x_probs"] = x_probs
    decoder_data["z_probs"] = z_probs
    decoder_data["x_faults_map"] = x_faults_map
    decoder_data["z_faults_map"] = z_faults_map
    decoder_data["cycle"] = cycle
    decoder_data["HX"] = HX
    decoder_data["HZ"] = HZ
    decoder_data["lx"] = lx
    decoder_data["lz"] = lz
    decoder_data["hx"] = hx
    decoder_data["hz"] = hz
    decoder_data["z_logical_row"] = z_logical_row
    decoder_data["x_logical_row"] = x_logical_row
    decoder_data["ell"] = ell
    decoder_data["m"] = m
    decoder_data["a1"] = a1
    decoder_data["a2"] = a2
    decoder_data["a3"] = a3
    decoder_data["b1"] = b1
    decoder_data["b2"] = b2
    decoder_data["b3"] = b3
    decoder_data["noise_model"] = noise_model
    decoder_data["x_check_coupling"] = x_check_coupling
    decoder_data["z_check_coupling"] = z_check_coupling
    print("Done.")
    return decoder_data


def amend_decoder_in_place(decoder_data: Dict, s_decoder_type: str, faults_to_add):
    """
    Amend decoder by adding new fault column combinations in-place.
    
    Extends the parity check matrix with columns representing specific multi-fault
    syndromes, improving decoder performance on low-weight errors.
    
    Args:
        decoder_data: Decoder data dictionary to modify.
        s_decoder_type: "x" or "z" to specify which decoder to amend.
        faults_to_add: List of fault column index combinations to add.
    """
    if s_decoder_type == "z":
        s_H = "HZ"
        s_H_decoder = "HZ_decoder"
        s_faults_map = "z_faults_map"
        s_probs = "z_probs"
    elif s_decoder_type == "x":
        s_H = "HX"
        s_H_decoder = "HX_decoder"
        s_faults_map = "x_faults_map"
        s_probs = "x_probs"
    else:
        raise Exception("Unknown decoder type")
    H = decoder_data[s_H]
    H_decoder = decoder_data[s_H_decoder]
    faults_map = decoder_data[s_faults_map]
    probs = decoder_data[s_probs]
    for fault in faults_to_add:
        h_columns = 0 * H.getcol(0).toarray().flatten()
        decoder_columns = 0 * H_decoder.getcol(0).toarray().flatten()
        p = 1.0
        for fault_col in fault:
            h_col = H.getcol(fault_col).toarray().flatten()
            decoder_col = H_decoder.getcol(fault_col).toarray().flatten()
            p *= probs[fault_col]
            h_columns = (h_columns + h_col) % 2
            decoder_columns = (decoder_columns + decoder_col) % 2
        H = hstack([H, h_columns[:, None]])
        H_decoder = hstack([H_decoder, decoder_columns[:, None]])
        probs.append(p)

    decoder_data[s_H] = H
    decoder_data[s_H_decoder] = H_decoder
    decoder_data[s_faults_map] = faults_map
    decoder_data[s_probs] = probs
