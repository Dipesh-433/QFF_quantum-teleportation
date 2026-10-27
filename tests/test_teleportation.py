"""Unit and integration test suite for Quantum Teleportation project.

Tests:
  - Input parameter validation and angle handling
  - State preparation accuracy for all five benchmark test states
  - Circuit synthesis (qubit count, classical bit count, dynamic instructions)
  - Ideal AerSimulator execution with verification success near 100%
  - Ideal state fidelity near 1.0 using Qiskit state_fidelity
  - Realistic noisy simulation degradation under depolarizing and readout noise
  - Full-state tomography density matrix reconstruction
  - Artifact and visualization generation (circuit_diagram, measurement, fidelity, CSV)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity

from src.fidelity import (
    evaluate_state,
    perform_quantum_state_tomography,
    reconstruct_density_matrix_from_pauli,
)
from src.simulator import NoiseConfig, QuantumSimulator
from src.state_preparation import (
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


class TestStatePreparation:
    """Tests covering state preparation routines and test states."""

    def test_angle_validation_valid(self):
        """Verify valid angle inputs pass through correctly."""
        th, ph = validate_angles(math.pi / 2, math.pi / 4)
        assert pytest.approx(th) == math.pi / 2
        assert pytest.approx(ph) == math.pi / 4

    def test_angle_validation_invalid_types(self):
        """Verify invalid types raise TypeError."""
        with pytest.raises(TypeError):
            validate_angles("pi", 0.0)
        with pytest.raises(TypeError):
            validate_angles(0.0, None)

    def test_angle_validation_nan_inf(self):
        """Verify NaN and Inf values raise ValueError."""
        with pytest.raises(ValueError):
            validate_angles(float("nan"), 0.0)
        with pytest.raises(ValueError):
            validate_angles(0.0, float("inf"))

    def test_all_five_test_states_defined(self):
        """Verify all five mandatory test states exist with correct metadata."""
        states = get_test_states()
        expected_keys = {"state_0", "state_1", "state_plus", "state_plus_i", "state_arbitrary"}
        assert set(states.keys()) == expected_keys

        for key, s in states.items():
            assert s.name
            assert s.symbol
            assert isinstance(s.theta, float)
            assert isinstance(s.phi, float)

    def test_state_preparation_ground_state(self):
        """Verify |0> state preparation produces exact |0> statevector."""
        sv = compute_theoretical_statevector(0.0, 0.0)
        prob_0 = np.abs(sv.data[0]) ** 2
        prob_1 = np.abs(sv.data[1]) ** 2
        assert pytest.approx(prob_0, abs=1e-6) == 1.0
        assert pytest.approx(prob_1, abs=1e-6) == 0.0

    def test_state_preparation_excited_state(self):
        """Verify |1> state preparation produces exact |1> statevector."""
        sv = compute_theoretical_statevector(math.pi, 0.0)
        prob_0 = np.abs(sv.data[0]) ** 2
        prob_1 = np.abs(sv.data[1]) ** 2
        assert pytest.approx(prob_0, abs=1e-6) == 0.0
        assert pytest.approx(prob_1, abs=1e-6) == 1.0

    def test_state_preparation_plus_state(self):
        """Verify |+> state preparation produces equal superposition."""
        sv = compute_theoretical_statevector(math.pi / 2.0, 0.0)
        prob_0 = np.abs(sv.data[0]) ** 2
        prob_1 = np.abs(sv.data[1]) ** 2
        assert pytest.approx(prob_0, abs=1e-6) == 0.5
        assert pytest.approx(prob_1, abs=1e-6) == 0.5

    def test_state_preparation_plus_i_state(self):
        """Verify |+i> state preparation produces phase superposition."""
        sv = compute_theoretical_statevector(math.pi / 2.0, math.pi / 2.0)
        prob_0 = np.abs(sv.data[0]) ** 2
        prob_1 = np.abs(sv.data[1]) ** 2
        assert pytest.approx(prob_0, abs=1e-6) == 0.5
        assert pytest.approx(prob_1, abs=1e-6) == 0.5
        # Phase ratio: c1 / c0 should equal exp(i * pi/2) = i
        phase_ratio = sv.data[1] / sv.data[0]
        assert pytest.approx(np.real(phase_ratio), abs=1e-6) == 0.0
        assert pytest.approx(np.imag(phase_ratio), abs=1e-6) == 1.0

    def test_inverse_state_preparation_reversibility(self):
        r"""Verify U^\dagger U = I maps prepared state back to |0> for all test states."""
        for state_key, s in get_test_states().items():
            qc = QuantumCircuit(1)
            apply_state_preparation(qc, 0, s.theta, s.phi)
            apply_inverse_state_preparation(qc, 0, s.theta, s.phi)
            sv = Statevector.from_instruction(qc)
            assert pytest.approx(np.abs(sv.data[0]) ** 2, abs=1e-6) == 1.0


class TestTeleportationCircuit:
    """Tests covering circuit synthesis and structure."""

    def test_circuit_qubit_and_clbit_counts(self):
        """Verify 3 qubits and expected classical register allocations."""
        qc_verif = build_teleportation_circuit(0.5, 0.5, include_verification=True)
        assert qc_verif.num_qubits == 3
        # cr_alice (2) + cr_bob (1) = 3 classical bits
        assert qc_verif.num_clbits == 3

        qc_no_verif = build_teleportation_circuit(0.5, 0.5, include_verification=False)
        assert qc_no_verif.num_qubits == 3
        assert qc_no_verif.num_clbits == 2

    def test_dynamic_feedforward_instructions(self):
        """Verify circuit contains if_test conditional branch operations."""
        qc = build_teleportation_circuit(0.5, 0.5, include_verification=True)
        op_names = [inst.operation.name for inst in qc.data]
        assert "if_else" in op_names, "Dynamic control flow (if_else) not present in circuit."

    def test_circuit_drawing_generates_figure(self, tmp_path: Path):
        """Verify circuit drawer generates a non-empty figure and saves PNG."""
        qc = build_teleportation_circuit(0.5, 0.5, include_verification=True)
        out_file = tmp_path / "test_circuit.png"
        fig = draw_circuit_diagram(qc, output_path=str(out_file), dpi=100)
        assert fig is not None
        assert out_file.exists()
        assert out_file.stat().st_size > 0


class TestSimulationAndFidelity:
    """Tests covering ideal execution, noisy simulation, and fidelity analysis."""

    @pytest.fixture(scope="class")
    @classmethod
    def simulator(cls):
        return QuantumSimulator(default_shots=2048, seed=42)

    @pytest.fixture(scope="class")
    @classmethod
    def noise_cfg(cls):
        return NoiseConfig(p_single_qubit=0.005, p_two_qubit=0.015, p_readout=0.02)

    @pytest.mark.parametrize("state_key", ["state_0", "state_1", "state_plus", "state_plus_i", "state_arbitrary"])
    def test_ideal_verification_success_near_100_percent(self, simulator, state_key):
        """Verify ideal simulation achieves ~100% verification success for all states."""
        s = get_test_states()[state_key]
        qc = build_teleportation_circuit(s.theta, s.phi, include_verification=True)
        res = simulator.run_ideal(qc, shots=2048)
        # Ideal verification success should equal 1.0 (all 2048 shots yield |0>)
        assert res.verification_success_probability == 1.0

    @pytest.mark.parametrize("state_key", ["state_0", "state_1", "state_plus", "state_plus_i", "state_arbitrary"])
    def test_ideal_state_fidelity_near_one(self, simulator, state_key):
        """Verify full quantum state tomography yields fidelity >= 0.97 under ideal shots."""
        s = get_test_states()[state_key]
        _, fidelity = perform_quantum_state_tomography(
            theta=s.theta,
            phi=s.phi,
            simulator=simulator,
            noise_config=None,
            shots=2048,
        )
        assert fidelity >= 0.97, f"Fidelity {fidelity} was below 0.97 for {s.name}"

    def test_noisy_simulation_shows_nonzero_error(self, simulator, noise_cfg):
        """Verify realistic noise degrades verification success below ideal 1.0."""
        s = get_test_states()["state_arbitrary"]
        qc = build_teleportation_circuit(s.theta, s.phi, include_verification=True)

        res_noisy = simulator.run_noisy(qc, noise_config=noise_cfg, shots=2048)
        # In noisy conditions, error rate should be > 0 and success < 1.0
        assert res_noisy.verification_success_probability < 1.0
        assert res_noisy.verification_success_probability >= 0.80

    def test_reconstruct_density_matrix_physical_validity(self):
        """Verify reconstructed density matrix is Hermitian, positive semi-definite with Tr=1."""
        # Test with random expectation values in [-1, 1]
        dm = reconstruct_density_matrix_from_pauli(exp_x=0.4, exp_y=-0.3, exp_z=0.5)
        mat = dm.data
        # Hermiticity
        assert np.allclose(mat, mat.conj().T)
        # Trace == 1
        assert pytest.approx(np.real(np.trace(mat)), abs=1e-6) == 1.0
        # Positive eigenvalues
        eigenvals = np.linalg.eigvalsh(mat)
        assert np.all(eigenvals >= -1e-7)

    def test_full_state_evaluation_report(self, simulator, noise_cfg):
        """Verify StateEvaluationReport computes all expected metrics."""
        s = get_test_states()["state_plus"]
        report = evaluate_state(s, simulator, noise_config=noise_cfg, shots=1024)
        assert report.ideal_verification_success == 1.0
        assert report.noisy_verification_success < 1.0
        assert report.noisy_verification_success > 0.80
        assert report.ideal_tomography_fidelity >= 0.95
        assert report.noisy_tomography_fidelity < report.ideal_tomography_fidelity
