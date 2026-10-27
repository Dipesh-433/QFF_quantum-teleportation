"""Simulation module for Quantum Teleportation.

This module provides reusable, high-performance simulation routines utilizing
Qiskit Aer (AerSimulator). It supports both ideal execution and realistic noisy
execution with configurable depolarizing gate errors and readout errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError


@dataclass
class NoiseConfig:
    """Configuration container for quantum noise simulation parameters."""

    p_single_qubit: float = 0.005  # 0.5% depolarizing error on 1-qubit gates
    p_two_qubit: float = 0.015     # 1.5% depolarizing error on 2-qubit gates (CX)
    p_readout: float = 0.020       # 2.0% measurement readout bit-flip error

    def build_noise_model(self) -> NoiseModel:
        """Construct a Qiskit Aer NoiseModel based on the configured error rates.

        Returns:
            Configured NoiseModel instance.
        """
        noise_model = NoiseModel()

        # 1-qubit depolarizing error for common single-qubit gates
        if self.p_single_qubit > 0.0:
            err_1q = depolarizing_error(self.p_single_qubit, 1)
            noise_model.add_all_qubit_quantum_error(
                err_1q,
                ["h", "ry", "rz", "x", "z", "s", "sdg", "sx"],
            )

        # 2-qubit depolarizing error for CNOT
        if self.p_two_qubit > 0.0:
            err_2q = depolarizing_error(self.p_two_qubit, 2)
            noise_model.add_all_qubit_quantum_error(err_2q, ["cx"])

        # Readout bit-flip error on measurement
        if self.p_readout > 0.0:
            p_ro = self.p_readout
            ro_error = ReadoutError([[1.0 - p_ro, p_ro], [p_ro, 1.0 - p_ro]])
            noise_model.add_all_qubit_readout_error(ro_error)

        return noise_model


@dataclass
class SimulationResult:
    """Detailed summary of a quantum circuit execution."""

    raw_counts: Dict[str, int]
    shots: int
    is_noisy: bool
    noise_config: Optional[NoiseConfig] = None
    seed: Optional[int] = None

    @property
    def total_shots(self) -> int:
        """Actual total shot count aggregated from raw counts."""
        return sum(self.raw_counts.values())

    @property
    def bob_counts(self) -> Dict[str, int]:
        """Aggregate counts for Bob's measurement bit (0 or 1).

        In Qiskit register bitstrings formatted as 'c_bob c_alice' (e.g., '0 01'):
        Bob's register is the first token. If only a single string without spaces is
        returned, Bob's bit is the leftmost character.
        """
        bob_dict = {"0": 0, "1": 0}
        for bitstring, count in self.raw_counts.items():
            token = bitstring.split()[0] if " " in bitstring else bitstring[0]
            bob_bit = token[0]
            bob_dict[bob_bit] = bob_dict.get(bob_bit, 0) + count
        return bob_dict

    @property
    def alice_counts(self) -> Dict[str, int]:
        """Aggregate counts for Alice's Bell-measurement outcomes ('00', '01', '10', '11')."""
        alice_dict: Dict[str, int] = {}
        for bitstring, count in self.raw_counts.items():
            if " " in bitstring:
                token = bitstring.split()[1]
            else:
                token = bitstring[1:] if len(bitstring) > 1 else ""
            alice_dict[token] = alice_dict.get(token, 0) + count
        return alice_dict

    @property
    def verification_success_probability(self) -> float:
        """Empirical probability of measuring Bob's qubit as |0> after inverse preparation."""
        total = self.total_shots
        if total == 0:
            return 0.0
        return self.bob_counts.get("0", 0) / float(total)

    @property
    def verification_error_rate(self) -> float:
        """Empirical error probability (1 - verification_success_probability)."""
        return 1.0 - self.verification_success_probability


class QuantumSimulator:
    """Wrapper around Qiskit AerSimulator supporting reproducible ideal and noisy runs."""

    def __init__(
        self,
        default_shots: int = 4096,
        seed: Optional[int] = 42,
    ) -> None:
        """Initialize the simulator.

        Args:
            default_shots: Default number of simulation shots.
            seed: Fixed random seed for deterministic reproducibility.
        """
        self.default_shots = default_shots
        self.seed = seed
        self._ideal_backend = AerSimulator()

    def run_ideal(
        self,
        circuit: QuantumCircuit,
        shots: Optional[int] = None,
    ) -> SimulationResult:
        """Execute a circuit under ideal (noise-free) conditions.

        Args:
            circuit: QuantumCircuit to execute.
            shots: Number of shots (defaults to self.default_shots).

        Returns:
            SimulationResult containing counts and summary metrics.
        """
        shot_count = shots if shots is not None else self.default_shots
        transpiled_circuit = qiskit.transpile(circuit, self._ideal_backend)

        job = self._ideal_backend.run(
            transpiled_circuit,
            shots=shot_count,
            seed_simulator=self.seed,
        )
        result = job.result()
        raw_counts = result.get_counts()

        return SimulationResult(
            raw_counts=raw_counts,
            shots=shot_count,
            is_noisy=False,
            seed=self.seed,
        )

    def run_noisy(
        self,
        circuit: QuantumCircuit,
        noise_config: Optional[NoiseConfig] = None,
        shots: Optional[int] = None,
    ) -> SimulationResult:
        """Execute a circuit under realistic depolarizing and readout noise.

        Args:
            circuit: QuantumCircuit to execute.
            noise_config: Custom NoiseConfig instance (or default parameters).
            shots: Number of shots.

        Returns:
            SimulationResult containing noisy counts and metrics.
        """
        cfg = noise_config if noise_config is not None else NoiseConfig()
        noise_model = cfg.build_noise_model()
        noisy_backend = AerSimulator(noise_model=noise_model)

        shot_count = shots if shots is not None else self.default_shots
        transpiled_circuit = qiskit.transpile(circuit, noisy_backend)

        job = noisy_backend.run(
            transpiled_circuit,
            shots=shot_count,
            seed_simulator=self.seed,
        )
        result = job.result()
        raw_counts = result.get_counts()

        return SimulationResult(
            raw_counts=raw_counts,
            shots=shot_count,
            is_noisy=True,
            noise_config=cfg,
            seed=self.seed,
        )
