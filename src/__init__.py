"""Quantum Teleportation package for Qiskit Fall Fest 2026.

This package provides a modular, tested implementation of the quantum teleportation
protocol using modern Qiskit Dynamic Circuits and Qiskit Aer simulation.
"""

from src.state_preparation import (
    QuantumStateInfo,
    apply_inverse_state_preparation,
    apply_state_preparation,
    compute_theoretical_density_matrix,
    compute_theoretical_statevector,
    get_test_states,
    validate_angles,
)
from src.teleportation_circuit import (
    build_teleportation_circuit,
    draw_circuit_diagram,
)
from src.simulator import (
    NoiseConfig,
    QuantumSimulator,
    SimulationResult,
)
from src.fidelity import (
    StateEvaluationReport,
    benchmark_reports_to_dataframe,
    evaluate_state,
    perform_quantum_state_tomography,
    reconstruct_density_matrix_from_pauli,
    run_benchmark_suite,
)

__all__ = [
    # State preparation
    "QuantumStateInfo",
    "apply_state_preparation",
    "apply_inverse_state_preparation",
    "get_test_states",
    "validate_angles",
    "compute_theoretical_statevector",
    "compute_theoretical_density_matrix",
    # Teleportation circuit
    "build_teleportation_circuit",
    "draw_circuit_diagram",
    # Simulator
    "NoiseConfig",
    "QuantumSimulator",
    "SimulationResult",
    # Fidelity & verification
    "StateEvaluationReport",
    "evaluate_state",
    "perform_quantum_state_tomography",
    "reconstruct_density_matrix_from_pauli",
    "run_benchmark_suite",
    "benchmark_reports_to_dataframe",
]

__version__ = "1.0.0"
