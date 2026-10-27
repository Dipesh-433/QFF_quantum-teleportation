"""State preparation module for Quantum Teleportation.

This module provides functions to prepare normalized single-qubit quantum states
using standard Qiskit unitary rotation gates, based on the convention:

    |psi> = Rz(phi) Ry(theta) |0>

Under this convention:
    Ry(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>
    Rz(phi) = exp(-i*phi*Z/2) = diag(exp(-i*phi/2), exp(i*phi/2))
    |psi> = exp(-i*phi/2) [ cos(theta/2)|0> + exp(i*phi) sin(theta/2)|1> ]

Ignoring the global phase exp(-i*phi/2), this parameterizes any arbitrary
single-qubit state on the Bloch sphere with polar angle theta in [0, pi] and
azimuthal angle phi in [0, 2*pi).

Standard test states:
    - |0>:   theta = 0,       phi = 0
    - |1>:   theta = pi,      phi = 0
    - |+>:   theta = pi/2,    phi = 0
    - |+i>:  theta = pi/2,    phi = pi/2
    - Arbitrary |psi>: theta = pi/3, phi = pi/4 (or user-defined)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.quantum_info import DensityMatrix, Statevector


@dataclass(frozen=True)
class QuantumStateInfo:
    """Metadata container for a benchmark quantum test state."""

    name: str
    symbol: str
    theta: float
    phi: float
    description: str

    @property
    def theoretical_statevector(self) -> Statevector:
        """Compute the exact theoretical Statevector under the Rz(phi)Ry(theta) convention."""
        qc = QuantumCircuit(1)
        apply_state_preparation(qc, 0, self.theta, self.phi)
        return Statevector.from_instruction(qc)

    @property
    def theoretical_density_matrix(self) -> DensityMatrix:
        """Compute the exact theoretical DensityMatrix."""
        return DensityMatrix(self.theoretical_statevector)


def validate_angles(theta: float, phi: float) -> Tuple[float, float]:
    """Validate and normalize rotation angles theta and phi.

    Args:
        theta: Polar angle (radians). Expected in [0, pi] or will be modulo reduced.
        phi: Azimuthal angle (radians). Expected in [0, 2*pi) or will be modulo reduced.

    Returns:
        Tuple of (theta, phi) as normalized floats.

    Raises:
        TypeError: If theta or phi is not a real numerical value.
        ValueError: If theta or phi is NaN or infinite.
    """
    if not isinstance(theta, (int, float, np.floating)):
        raise TypeError(f"theta must be a real number, got {type(theta).__name__}")
    if not isinstance(phi, (int, float, np.floating)):
        raise TypeError(f"phi must be a real number, got {type(phi).__name__}")

    theta_val = float(theta)
    phi_val = float(phi)

    if math.isnan(theta_val) or math.isinf(theta_val):
        raise ValueError(f"Invalid theta value: {theta_val}")
    if math.isnan(phi_val) or math.isinf(phi_val):
        raise ValueError(f"Invalid phi value: {phi_val}")

    return theta_val, phi_val


def apply_state_preparation(
    circuit: QuantumCircuit,
    qubit: Union[int, Qubit],
    theta: float,
    phi: float,
) -> QuantumCircuit:
    """Apply unitary rotation gates Ry(theta) then Rz(phi) to prepare |psi>.

    Convention:
        |psi> = Rz(phi) Ry(theta) |0>

    Args:
        circuit: Target Qiskit QuantumCircuit.
        qubit: Qubit index or Qubit object to prepare.
        theta: Polar rotation angle in radians.
        phi: Azimuthal rotation angle in radians.

    Returns:
        The modified QuantumCircuit (for chaining).
    """
    theta_val, phi_val = validate_angles(theta, phi)
    circuit.ry(theta_val, qubit)
    circuit.rz(phi_val, qubit)
    return circuit


def apply_inverse_state_preparation(
    circuit: QuantumCircuit,
    qubit: Union[int, Qubit],
    theta: float,
    phi: float,
) -> QuantumCircuit:
    """Apply inverse unitary gates U^\dagger = Ry(-theta) Rz(-phi) to verify |psi>.

    Since U = Rz(phi) Ry(theta), its adjoint is:
        U^dagger = Ry(-theta) Rz(-phi)
    Applying U^dagger to |psi> maps it back to |0>, allowing a computational basis
    measurement to serve as an exact projective verification witness.

    Args:
        circuit: Target Qiskit QuantumCircuit.
        qubit: Qubit index or Qubit object on which to apply inverse gates.
        theta: Polar angle used in the forward preparation.
        phi: Azimuthal angle used in the forward preparation.

    Returns:
        The modified QuantumCircuit.
    """
    theta_val, phi_val = validate_angles(theta, phi)
    circuit.rz(-phi_val, qubit)
    circuit.ry(-theta_val, qubit)
    return circuit


def get_test_states() -> Dict[str, QuantumStateInfo]:
    """Retrieve the standard suite of five benchmark test states.

    Returns:
        Dictionary mapping state identifiers to QuantumStateInfo instances:
            - 'state_0':  |0>
            - 'state_1':  |1>
            - 'state_plus': |+>
            - 'state_plus_i': |+i>
            - 'state_arbitrary': Arbitrary Bloch sphere state (theta=pi/3, phi=pi/4)
    """
    return {
        "state_0": QuantumStateInfo(
            name="Ground State |0>",
            symbol="|0>",
            theta=0.0,
            phi=0.0,
            description="Computational ground basis state |0> (north pole of Bloch sphere).",
        ),
        "state_1": QuantumStateInfo(
            name="Excited State |1>",
            symbol="|1>",
            theta=math.pi,
            phi=0.0,
            description="Computational excited basis state |1> (south pole of Bloch sphere).",
        ),
        "state_plus": QuantumStateInfo(
            name="Equal Superposition |+>",
            symbol="|+>",
            theta=math.pi / 2.0,
            phi=0.0,
            description="Equal superposition (|0> + |1>)/sqrt(2) (+X axis on Bloch sphere).",
        ),
        "state_plus_i": QuantumStateInfo(
            name="Phase Superposition |+i>",
            symbol="|+i>",
            theta=math.pi / 2.0,
            phi=math.pi / 2.0,
            description="Equal superposition with complex phase (|0> + i|1>)/sqrt(2) (+Y axis).",
        ),
        "state_arbitrary": QuantumStateInfo(
            name="Arbitrary State |psi>",
            symbol="|psi>",
            theta=math.pi / 3.0,  # 60 degrees
            phi=math.pi / 4.0,   # 45 degrees
            description="Arbitrary superposition with non-trivial polar and azimuthal phases.",
        ),
    }


def compute_theoretical_statevector(theta: float, phi: float) -> Statevector:
    """Compute the exact single-qubit statevector for given theta and phi.

    Args:
        theta: Polar angle in radians.
        phi: Azimuthal angle in radians.

    Returns:
        Qiskit Statevector instance representing |psi>.
    """
    qc = QuantumCircuit(1)
    apply_state_preparation(qc, 0, theta, phi)
    return Statevector.from_instruction(qc)


def compute_theoretical_density_matrix(theta: float, phi: float) -> DensityMatrix:
    """Compute the exact single-qubit density matrix for given theta and phi.

    Args:
        theta: Polar angle in radians.
        phi: Azimuthal angle in radians.

    Returns:
        Qiskit DensityMatrix instance representing rho = |psi><psi|.
    """
    return DensityMatrix(compute_theoretical_statevector(theta, phi))
