"""Fidelity and quantum state verification module for Quantum Teleportation.

This module implements mathematically rigorous state verification, quantum state
tomography (QST), and fidelity analysis.

Scientific Distinctions & Methodology:
--------------------------------------
1. Verification Success Probability P(|0>):
   When Bob receives his qubit, we apply the inverse unitary transformation
   U^\dagger = Ry(-theta) Rz(-phi) and measure in the computational basis.
   The probability of measuring |0> equals:
       P(|0>) = <0| U^\dagger rho_Bob U |0> = <psi| rho_Bob |psi>
   For a pure target state |psi>, this equals the project-onto-target fidelity.
   However, as an experimental estimator, it relies on a single projective measurement
   axis and is directly vulnerable to asymmetric readout errors. It does NOT
   characterize the full rank-2 density matrix.

2. Full Quantum State Tomography (QST):
   To characterize Bob's reconstructed state without assuming purity or alignment,
   we execute the teleportation protocol with measurement along three mutually
   unbiased Pauli bases {X, Y, Z}:
       <X> = P(0_X) - P(1_X)  [measured after H gate]
       <Y> = P(0_Y) - P(1_Y)  [measured after S^\dagger H gate]
       <Z> = P(0_Z) - P(1_Z)  [measured in computational basis]
   The empirical single-qubit density matrix is reconstructed via:
       rho_rec = (1/2) * [ I + <X> sigma_x + <Y> sigma_y + <Z> sigma_z ]
   We enforce physical validity (Hermitian, positive semi-definite with Tr(rho)=1)
   by clipping negative eigenvalues and renormalizing.

3. Quantum State Fidelity:
   Using Jozsa's generalization of Uhlmann's fidelity:
       F(rho_target, rho_rec) = ( Tr sqrt( sqrt(rho_target) rho_rec sqrt(rho_target) ) )^2
   evaluated via `qiskit.quantum_info.state_fidelity`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity

from src.simulator import NoiseConfig, QuantumSimulator, SimulationResult
from src.state_preparation import QuantumStateInfo, get_test_states
from src.teleportation_circuit import build_teleportation_circuit


# Pauli matrices and identity
_SIGMA_I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
_SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


@dataclass
class StateEvaluationReport:
    """Consolidated metrics for a single quantum test state under ideal and noisy simulation."""

    state_info: QuantumStateInfo
    ideal_verification_success: float
    noisy_verification_success: float
    ideal_tomography_fidelity: float
    noisy_tomography_fidelity: float
    ideal_rho_rec: DensityMatrix
    noisy_rho_rec: DensityMatrix
    target_rho: DensityMatrix


def reconstruct_density_matrix_from_pauli(
    exp_x: float,
    exp_y: float,
    exp_z: float,
) -> DensityMatrix:
    """Construct a physical, trace-normalized single-qubit DensityMatrix from Pauli expectations.

    Args:
        exp_x: Expectation value <X> in [-1, 1].
        exp_y: Expectation value <Y> in [-1, 1].
        exp_z: Expectation value <Z> in [-1, 1].

    Returns:
        Valid Qiskit DensityMatrix instance.
    """
    raw_matrix = 0.5 * (
        _SIGMA_I
        + exp_x * _SIGMA_X
        + exp_y * _SIGMA_Y
        + exp_z * _SIGMA_Z
    )

    # Ensure Hermiticity
    hermitian_matrix = 0.5 * (raw_matrix + raw_matrix.conj().T)

    # Diagonalize and enforce positive semi-definiteness (eigenvalue clipping)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian_matrix)
    clipped_eigenvalues = np.maximum(eigenvalues, 0.0)
    trace = np.sum(clipped_eigenvalues)

    if trace > 0.0:
        normalized_eigenvalues = clipped_eigenvalues / trace
    else:
        normalized_eigenvalues = np.array([0.5, 0.5])

    valid_matrix = eigenvectors @ np.diag(normalized_eigenvalues) @ eigenvectors.conj().T
    return DensityMatrix(valid_matrix)


def perform_quantum_state_tomography(
    theta: float,
    phi: float,
    simulator: QuantumSimulator,
    noise_config: Optional[NoiseConfig] = None,
    shots: int = 4096,
) -> Tuple[DensityMatrix, float]:
    """Execute single-qubit quantum state tomography on Bob's teleported qubit.

    Args:
        theta: Polar angle of input state.
        phi: Azimuthal angle of input state.
        simulator: QuantumSimulator instance.
        noise_config: Noise configuration (None for ideal).
        shots: Shots per measurement basis.

    Returns:
        Tuple of (reconstructed_density_matrix, state_fidelity_against_target).
    """
    from src.state_preparation import compute_theoretical_density_matrix

    target_rho = compute_theoretical_density_matrix(theta, phi)

    exp_values = {}
    for basis in ["X", "Y", "Z"]:
        circuit = build_teleportation_circuit(
            theta=theta,
            phi=phi,
            include_verification=False,
            tomography_basis=basis,
        )
        if noise_config is None:
            sim_res = simulator.run_ideal(circuit, shots=shots)
        else:
            sim_res = simulator.run_noisy(circuit, noise_config=noise_config, shots=shots)

        bob_c = sim_res.bob_counts
        n0 = bob_c.get("0", 0)
        n1 = bob_c.get("1", 0)
        total = n0 + n1
        exp_val = (n0 - n1) / float(total) if total > 0 else 0.0
        exp_values[basis] = exp_val

    rho_rec = reconstruct_density_matrix_from_pauli(
        exp_values["X"],
        exp_values["Y"],
        exp_values["Z"],
    )
    fidelity = float(state_fidelity(target_rho, rho_rec))
    return rho_rec, fidelity


def evaluate_state(
    state_info: QuantumStateInfo,
    simulator: QuantumSimulator,
    noise_config: Optional[NoiseConfig] = None,
    shots: int = 4096,
) -> StateEvaluationReport:
    """Perform full evaluation of a test state: inverse verification and tomography.

    Args:
        state_info: Metadata for the state.
        simulator: Simulator instance.
        noise_config: Configurable noise settings.
        shots: Shots per circuit execution.

    Returns:
        StateEvaluationReport summarizing verification probabilities and fidelities.
    """
    noise_cfg = noise_config if noise_config is not None else NoiseConfig()
    target_rho = state_info.theoretical_density_matrix

    # 1. Inverse verification circuit
    verif_circuit = build_teleportation_circuit(
        theta=state_info.theta,
        phi=state_info.phi,
        include_verification=True,
    )

    # Run ideal verification
    res_ideal = simulator.run_ideal(verif_circuit, shots=shots)
    ideal_verif = res_ideal.verification_success_probability

    # Run noisy verification
    res_noisy = simulator.run_noisy(verif_circuit, noise_config=noise_cfg, shots=shots)
    noisy_verif = res_noisy.verification_success_probability

    # 2. Quantum State Tomography (ideal and noisy)
    ideal_rho_rec, ideal_tomog_fid = perform_quantum_state_tomography(
        theta=state_info.theta,
        phi=state_info.phi,
        simulator=simulator,
        noise_config=None,
        shots=shots,
    )

    noisy_rho_rec, noisy_tomog_fid = perform_quantum_state_tomography(
        theta=state_info.theta,
        phi=state_info.phi,
        simulator=simulator,
        noise_config=noise_cfg,
        shots=shots,
    )

    return StateEvaluationReport(
        state_info=state_info,
        ideal_verification_success=ideal_verif,
        noisy_verification_success=noisy_verif,
        ideal_tomography_fidelity=ideal_tomog_fid,
        noisy_tomography_fidelity=noisy_tomog_fid,
        ideal_rho_rec=ideal_rho_rec,
        noisy_rho_rec=noisy_rho_rec,
        target_rho=target_rho,
    )


def run_benchmark_suite(
    simulator: Optional[QuantumSimulator] = None,
    noise_config: Optional[NoiseConfig] = None,
    shots: int = 4096,
) -> List[StateEvaluationReport]:
    """Execute the full teleportation benchmark on all five test states.

    Args:
        simulator: QuantumSimulator instance (creates default if None).
        noise_config: NoiseConfig instance (creates default if None).
        shots: Number of shots per circuit run.

    Returns:
        List of StateEvaluationReport objects for all benchmark states.
    """
    sim = simulator if simulator is not None else QuantumSimulator()
    noise_cfg = noise_config if noise_config is not None else NoiseConfig()
    test_states = get_test_states()

    reports = []
    for key, state_info in test_states.items():
        report = evaluate_state(
            state_info=state_info,
            simulator=sim,
            noise_config=noise_cfg,
            shots=shots,
        )
        reports.append(report)

    return reports


def benchmark_reports_to_dataframe(reports: List[StateEvaluationReport]) -> pd.DataFrame:
    """Format benchmark evaluation reports into a clean pandas DataFrame.

    Args:
        reports: List of StateEvaluationReport instances.

    Returns:
        pandas DataFrame with tabular metrics.
    """
    rows = []
    for r in reports:
        rows.append(
            {
                "State Name": r.state_info.name,
                "Symbol": r.state_info.symbol,
                "Theta (rad)": round(r.state_info.theta, 4),
                "Phi (rad)": round(r.state_info.phi, 4),
                "Ideal Verif Success": round(r.ideal_verification_success, 4),
                "Noisy Verif Success": round(r.noisy_verification_success, 4),
                "Ideal Tomog Fidelity": round(r.ideal_tomography_fidelity, 4),
                "Noisy Tomog Fidelity": round(r.noisy_tomography_fidelity, 4),
            }
        )
    return pd.DataFrame(rows)
