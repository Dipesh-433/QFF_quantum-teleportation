# Quantum Teleportation Using Qiskit

> **Qiskit Fall Fest 2026** | Dr. Babasaheb Ambedkar Technological University (DBATU), Lonere, Maharashtra, India

A complete, modular, tested, and GitHub-ready implementation of the **quantum teleportation protocol** using modern Qiskit Dynamic Circuits and Qiskit Aer simulation. This project implements genuine measurement-based classical feed-forward (not deferred measurement) and provides both projective verification and full quantum state tomography with Uhlmann-Jozsa fidelity analysis.

---

## Overview

Quantum teleportation transfers an unknown quantum state |psi> from Alice to Bob using:

- A shared maximally entangled Bell pair |Phi+>
- Two classical bits of communication
- Conditional Pauli corrections (X and Z gates)

This project demonstrates the complete protocol with five benchmark quantum states, comparing ideal and noisy simulation results.

## Objectives

- Implement the 3-qubit quantum teleportation protocol with **actual measurement and classical feed-forward** using Qiskit's `if_test` dynamic circuit semantics
- Support five canonical test states: |0>, |1>, |+>, |+i>, and an arbitrary parameterized state
- Provide **dual verification methods**: inverse-operator projective witnessing and full 3-axis quantum state tomography
- Compare ideal noise-free and realistic noisy simulation environments
- Deliver publication-quality visualizations, automated pytest validation, and an academic PDF report

## Quantum Teleportation Workflow

```
1. Alice prepares input state:    |psi> = Rz(phi) Ry(theta) |0>
2. Shared Bell pair created:      |Phi+> = (|00> + |11>) / sqrt(2)
3. Alice entangles & measures:    CNOT(q0, q1) -> H(q0) -> Measure q0, q1
4. Classical communication:       Alice sends 2 bits (c0, c1) to Bob
5. Bob applies corrections:       X(q2) if c1=1, Z(q2) if c0=1
6. Bob recovers |psi> on q2:      State teleportation complete!
```

## Project Architecture

```text
quantum-teleportation/
|
+-- notebooks/
|   +-- quantum_teleportation.ipynb    # Educational Jupyter notebook
|
+-- src/
|   +-- __init__.py                    # Package exports
|   +-- state_preparation.py           # Parameterized state preparation (Ry, Rz)
|   +-- teleportation_circuit.py       # Dynamic feed-forward circuit builder
|   +-- simulator.py                   # AerSimulator wrapper (ideal + noisy)
|   +-- fidelity.py                    # Verification + QST + state fidelity
|
+-- results/
|   +-- circuit_diagram.png            # Teleportation circuit visualization
|   +-- measurement_results.png        # Measurement outcome histograms
|   +-- fidelity_results.png           # Fidelity benchmark comparison
|   +-- results.csv                    # Numerical results table
|
+-- report/
|   +-- project_report.pdf             # Generated academic report
|
+-- tests/
|   +-- test_teleportation.py          # 25 automated test cases
|
+-- requirements.txt                   # Pinned dependencies
+-- README.md                          # This file
+-- main.py                            # CLI entry point (runs simulation, generates figures & PDF report)
+-- .gitignore                         # Git exclusions
```

## Prerequisites

- **Python 3.11** (tested with 3.11.7)
- **pip** package manager
- **Git** (for version control)

## Setup Instructions

### Windows (PowerShell)

```powershell
# Clone or navigate to the project directory
cd quantum-teleportation

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### macOS / Linux (Bash)

```bash
# Clone or navigate to the project directory
cd quantum-teleportation

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Main Application

From the project root directory with the virtual environment activated:

```bash
python main.py
```

This will:
1. Configure ideal and noisy AerSimulator backends
2. Build the quantum teleportation circuit with dynamic feed-forward
3. Execute benchmark simulations across all 5 test states
4. Generate visualization PNGs in `results/`
5. Export numerical results to `results/results.csv`
6. Compile the official academic report in `report/project_report.pdf`
7. Print a formatted results summary table

### Launching the Jupyter Notebook

```bash
jupyter notebook notebooks/quantum_teleportation.ipynb
```

Or from the project root:

```bash
cd quantum-teleportation
jupyter notebook notebooks/quantum_teleportation.ipynb
```

The notebook imports code from `src/` and must be launched from the project root directory.

### Running Tests

```bash
pytest -v
```

All 25 tests should pass, covering:
- Input validation and angle handling
- State preparation accuracy for all 5 states
- Circuit structure (qubit/clbit counts, dynamic instructions)
- Ideal verification success = 100%
- Ideal state fidelity >= 0.97
- Noisy degradation under configurable noise model
- Density matrix physical validity (Hermitian, PSD, Tr=1)


## Output Files

| File | Description |
|------|-------------|
| `results/circuit_diagram.png` | Publication-quality diagram of the teleportation circuit with measurements, classical registers, and conditional corrections |
| `results/measurement_results.png` | Dual-panel histogram: Alice's Bell-measurement bitstrings (4 outcomes) and Bob's verification readout |
| `results/fidelity_results.png` | Side-by-side benchmark: projective verification success vs. full QST fidelity for all 5 states |
| `results/results.csv` | Numerical results table with theta, phi, ideal/noisy verification, and tomography fidelities |
| `report/project_report.pdf` | Complete academic report with theory, methodology, results, and references |

## Results Summary

Results obtained from actual Qiskit Aer simulations (4,096 shots, seed=42):

| State | Symbol | Ideal Verification | Noisy Verification | Ideal Tomog. Fidelity | Noisy Tomog. Fidelity |
|-------|--------|-------------------|-------------------|----------------------|----------------------|
| Ground State | \|0> | 100.0% | 94.3% | 0.9996 | 0.9426 |
| Excited State | \|1> | 100.0% | 93.4% | 0.9996 | 0.9377 |
| Equal Superposition | \|+> | 100.0% | 92.9% | 0.9996 | 0.9294 |
| Phase Superposition | \|+i> | 100.0% | 91.2% | 0.9996 | 0.9119 |
| Arbitrary State | \|psi> | 100.0% | 92.5% | 0.9819 | 0.9214 |

### Key Observations

- **Ideal simulation** achieves exactly 100.0% verification success for all states, confirming correct protocol implementation
- **Noisy simulation** degrades gracefully to 91-94% under realistic depolarizing gate noise (0.5% single-qubit, 1.5% CX) and 2% readout errors
- **Full quantum state tomography** provides fidelities consistent with verification measurements, validating both assessment methods
- All results are **simulated** using Qiskit Aer; no real IBM Quantum hardware was used

## Noise Model Configuration

The project uses a configurable `NoiseConfig` dataclass:

```python
from src.simulator import NoiseConfig

noise_cfg = NoiseConfig(
    p_single_qubit=0.005,  # 0.5% depolarizing on H, Ry, Rz, X, Z
    p_two_qubit=0.015,     # 1.5% depolarizing on CX gates
    p_readout=0.020,       # 2.0% measurement readout bit-flip
)
```

## Verification vs. Fidelity: Scientific Distinction

This project carefully distinguishes two metrics:

1. **Projective Verification Success P(|0>)**: Applies U-dagger to Bob's qubit and measures in the Z basis. Equals the overlap <psi|rho_Bob|psi>. Simple but relies on prior knowledge of the target state.

2. **Full Quantum State Fidelity F(rho_target, rho_rec)**: Reconstructs Bob's density matrix via 3-axis Pauli tomography (X, Y, Z measurements), then computes the Uhlmann-Jozsa fidelity. Assumption-free and provides complete state characterization.

## Limitations

- All simulations are performed on classical hardware using Qiskit Aer; results are not from real quantum processors
- Noise model uses simplified symmetric depolarizing channels; real hardware exhibits asymmetric, correlated, and time-varying noise
- Single-qubit tomography uses linear inversion with eigenvalue clipping; maximum likelihood estimation (MLE) would improve accuracy at low shot counts
- No quantum error mitigation (ZNE, TREX) was applied

## Future Improvements

- Execute on real IBM Quantum hardware via Qiskit Runtime
- Apply quantum error mitigation techniques
- Extend to multi-qubit teleportation and entanglement swapping
- Implement maximum likelihood estimation (MLE) for tomography
- Add support for custom noise models from real device calibration data

## References

1. C. H. Bennett et al., "Teleporting an unknown quantum state via dual classical and EPR channels," *Phys. Rev. Lett.*, vol. 70, pp. 1895-1899, 1993.
2. M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information*, Cambridge University Press, 2010.
3. [Qiskit Documentation](https://docs.quantum.ibm.com/)
4. [Qiskit Aer Documentation](https://qiskit.github.io/qiskit-aer/)
5. [Qiskit Textbook - Quantum Teleportation](https://learning.quantum.ibm.com/)

## License

This project was developed for educational purposes as part of Qiskit Fall Fest 2026 at DBATU, Lonere.
