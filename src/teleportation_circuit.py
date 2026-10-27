"""Quantum Teleportation Circuit Module.

This module implements the canonical three-qubit quantum teleportation protocol
originally proposed by Bennett et al. (1993), using modern Qiskit Dynamic Circuit
primitives (actual measurement and conditional classical feed-forward via `if_test`).

Protocol Specification:
-----------------------
Qubits:
    - Qubit 0 (q0): Alice's input state (|psi> = Rz(phi) Ry(theta) |0>)
    - Qubit 1 (q1): Alice's half of the shared Bell pair (|Phi+>)
    - Qubit 2 (q2): Bob's half of the shared Bell pair (|Phi+>)

Classical Registers:
    - cr_alice (2 bits):
        - cr_alice[0]: Result of Alice's measurement on q0 (after Hadamard).
        - cr_alice[1]: Result of Alice's measurement on q1 (after CNOT).
    - cr_bob (1 bit, optional): Result of Bob's verification or state measurement.

Protocol Execution Sequence:
----------------------------
1. State Preparation:
   Apply Ry(theta) and Rz(phi) to q0.

2. Shared Entanglement Generation:
   Apply H to q1 and CX(q1, q2) to create |Phi+> = (|00> + |11>)/sqrt(2).

3. Bell-State Measurement (Alice's operations):
   Apply CX(q0, q1) followed by H(q0).
   This projects the joint state of q0 and q1 into the Bell basis.

4. Alice's Classical Measurement:
   Measure q0 -> cr_alice[0] (first protocol measurement bit).
   Measure q1 -> cr_alice[1] (second protocol measurement bit).

5. Classical Feed-Forward and Bob's Conditional Corrections:
   Communicate cr_alice to Bob.
   - If cr_alice[1] == 1 (Alice's second bit requires X flip):
       Apply Pauli-X to Bob's qubit (q2).
   - If cr_alice[0] == 1 (Alice's first bit requires Z flip):
       Apply Pauli-Z to Bob's qubit (q2).

   After correction, Bob's qubit q2 is guaranteed to be in state |psi>.

6. Verification / Readout:
   - For inverse-state verification: Apply U^\dagger = Ry(-theta) Rz(-phi) to q2,
     then measure q2 -> cr_bob[0]. A measurement of 0 confirms faithful teleportation.
   - For quantum state tomography: Measure q2 in Pauli X, Y, or Z bases.
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.visualization import circuit_drawer

from src.state_preparation import apply_inverse_state_preparation, apply_state_preparation


def build_teleportation_circuit(
    theta: float,
    phi: float,
    include_verification: bool = True,
    tomography_basis: Optional[str] = None,
    barrier: bool = True,
) -> QuantumCircuit:
    """Build the complete 3-qubit quantum teleportation circuit with feed-forward.

    Args:
        theta: Polar angle for Alice's input state |psi>.
        phi: Azimuthal angle for Alice's input state |psi>.
        include_verification: If True, appends inverse state preparation
            U^\dagger = Ry(-theta) Rz(-phi) to Bob's qubit and measures into cr_bob.
        tomography_basis: Optional basis for Bob's readout:
            - 'Z': Computational basis measurement
            - 'X': Measure after applying H gate
            - 'Y': Measure after applying S^\dagger followed by H gate
            - None: No tomography basis rotation (or governed by include_verification)
        barrier: Whether to insert visual barriers between protocol stages.

    Returns:
        Configured Qiskit QuantumCircuit.
    """
    qr = QuantumRegister(3, name="q")
    cr_alice = ClassicalRegister(2, name="c_alice")

    if include_verification or (tomography_basis is not None):
        cr_bob = ClassicalRegister(1, name="c_bob")
        circuit = QuantumCircuit(qr, cr_alice, cr_bob, name="quantum_teleportation")
    else:
        circuit = QuantumCircuit(qr, cr_alice, name="quantum_teleportation")

    # --- Stage 1: Alice prepares input state |psi> on q0 ---
    apply_state_preparation(circuit, qr[0], theta, phi)
    if barrier:
        circuit.barrier()

    # --- Stage 2: Create entangled Bell pair |Phi+> between q1 and q2 ---
    circuit.h(qr[1])
    circuit.cx(qr[1], qr[2])
    if barrier:
        circuit.barrier()

    # --- Stage 3: Alice entangles input q0 with Bell-pair qubit q1 ---
    circuit.cx(qr[0], qr[1])
    circuit.h(qr[0])
    if barrier:
        circuit.barrier()

    # --- Stage 4: Alice measures her two qubits ---
    circuit.measure(qr[0], cr_alice[0])
    circuit.measure(qr[1], cr_alice[1])
    if barrier:
        circuit.barrier()

    # --- Stage 5: Bob applies conditional Pauli corrections ---
    # Apply X when Alice's second measurement bit (cr_alice[1]) is 1
    with circuit.if_test((cr_alice[1], 1)):
        circuit.x(qr[2])

    # Apply Z when Alice's first measurement bit (cr_alice[0]) is 1
    with circuit.if_test((cr_alice[0], 1)):
        circuit.z(qr[2])

    # --- Stage 6: Readout / Verification on Bob's qubit (q2) ---
    if include_verification:
        if barrier:
            circuit.barrier()
        apply_inverse_state_preparation(circuit, qr[2], theta, phi)
        circuit.measure(qr[2], cr_bob[0])

    elif tomography_basis is not None:
        if barrier:
            circuit.barrier()
        basis = tomography_basis.upper()
        if basis == "X":
            circuit.h(qr[2])
        elif basis == "Y":
            circuit.sdg(qr[2])
            circuit.h(qr[2])
        elif basis == "Z":
            pass  # Computational basis directly
        else:
            raise ValueError(f"Unknown tomography basis '{tomography_basis}'. Choose 'X', 'Y', or 'Z'.")
        circuit.measure(qr[2], cr_bob[0])

    return circuit


def draw_circuit_diagram(
    circuit: QuantumCircuit,
    output_path: Optional[str] = None,
    style: str = "iqp",
    fold: int = -1,
    dpi: int = 300,
) -> plt.Figure:
    """Render and optionally save a publication-quality diagram of the circuit.

    Args:
        circuit: QuantumCircuit to visualize.
        output_path: Optional path to save PNG.
        style: Visualization style ('iqp', 'clifford', or dict).
        fold: Maximum circuit depth before line-wrapping. Default -1 disables folding.
        dpi: Resolution in dots per inch for saved figure.

    Returns:
        Matplotlib Figure object.
    """
    fig = circuit_drawer(circuit, output="mpl", style=style, fold=fold)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return fig
