# MATLAB Code for Stability Analysis of Distributed-Delay Systems

Supplementary MATLAB programs for the book chapter **“From Lyapunov Theory to Computation: A Practical Framework for Distributed Delay Systems”** by Alejandro Castaño Hernández.

This repository contains the two main MATLAB implementations discussed in the chapter and a separate driver for selected recurrence-level checks. It intentionally does **not** include the book-chapter manuscript, figures, parameter-map exports, result tables, or internal diagnostic archives.

## Repository structure

```text
src/
  Example1_ConstantKernel.m       % Scalar system with a constant kernel
  Example2_ExponentialKernel.m    % Matrix system with an exponential kernel
verification/
  validate_recurrences_matlab.m   % Selected coefficient-level checks
```

## Main programs

### Example 1 — Constant kernel

`src/Example1_ConstantKernel.m` implements the scalar constant-kernel example. Its default configuration is `analysisMode = 'exploration'`, with Legendre order `n = 5` and a 141-by-141 parameter grid. The script also contains an optional `criterion` mode with a conservative prescribed-order bound and a configurable order ceiling.

### Example 2 — Exponential kernel

`src/Example2_ExponentialKernel.m` implements the matrix-valued single-exponential-kernel example. Its default configuration is `analysisMode = 'exploration'`, with Legendre order `n = 5` and a 121-by-121 parameter grid. The script also contains an optional `criterion` mode with a conservative prescribed-order bound and a configurable order ceiling.

Both main programs use English comments, section headings, variable identifiers, diagnostic labels, and output-table names. Each execution creates timestamped output files in MATLAB's current working folder.

## Verification driver

Run `verification/validate_recurrences_matlab.m` separately to compare selected recurrence coefficients with direct Gauss–Legendre quadrature calculations. The driver writes three CSV files to the current MATLAB working folder:

- `matlab_constant_recurrence_checks.csv`
- `matlab_exponential_recurrence_checks.csv`
- `matlab_zero_mode_checks.csv`

These are selected formula-level checks. They do **not** execute either full parameter sweep, validate the complete assembled matrix `P_n`, or establish a stability region.

## Running the programs

1. Use MATLAB with support for scripts containing local functions.
2. Download or clone this repository.
3. Open MATLAB and set the current folder to the repository root.
4. Run either main program from the `src` folder, or use MATLAB's `run` command with the corresponding relative path. Run the verification driver separately when needed.
5. Retain the MATLAB release, operating system, command-window output, and generated CSV files when documenting a computational run.

The source uses standard MATLAB numerical functions including `expm`, `kron`, `eig`, `svd`, `rcond`, `chol`, and `writetable`. No additional toolbox dependency is explicitly identified in the source files. Runtime compatibility has not been verified against a specific MATLAB release in this repository snapshot.

## Important numerical-status notice

The parameter scans in the main programs are finite-order exploratory screens by default. A positive-definiteness result at order five is **not**, by itself, a theorem-level stability certificate. A conclusive application of the necessary-and-sufficient criterion requires the theorem's prescribed approximation order and verification of all relevant hypotheses.

The optional `criterion` modes estimate a conservative order bound and attempt the corresponding finite-dimensional test subject to a configured order ceiling. These modes have not been validated end to end in the supplied source snapshot.

For the exponential-kernel example, the current implementation includes block-symmetry diagnostics. The symmetry-residual issue documented during development remains unresolved in this snapshot. Its finite-order map must therefore be interpreted as a numerical diagnostic, not as an established stability region.

The recurrence verification driver checks selected coefficients only. Agreement at this level does not certify the full matrix assembly or the parameter-sweep results.

## Citation

Please cite the accompanying chapter when using these programs:

> Alejandro Castaño Hernández, “From Lyapunov Theory to Computation: A Practical Framework for Distributed Delay Systems,” IGI Global Scientific Publishing, chapter manuscript.

The repository includes a `CITATION.cff` file for GitHub's citation interface.

## License

No software license has been selected for this repository snapshot. Unless and until the author adds a license, no permission is granted to redistribute or modify the code beyond the rights provided by applicable law. Please contact the author for permission to reuse the programs.
