# Spline Integration

This repository contains the code required to reproduce the experimental results shown in the paper: *T. Teijeiro, A. Hashemian, J.M. Taylor and D. Pardo*, **Machine Learning Discovery of Optimal Quadrature Rules for Isogeometric Analysis**, 2023.

## Structure of the Repository:

- `splinequadrature.py`: Module defining the main functions implemented the method for quadrature rule discovery: Definition of the basis functions, loss function, parameter initialization, optimization, etc.
- `Spline Integration.ipynb`: Jupyter Notebook to interactively test the method and visualize the results.
- `optimal_uniform_rules.csv`: Contains the optimal quadrature rules, in double precision, for all uniform spline spaces with degree up to 16, and up to 50 elements. All possible continuity values are also considered. The file is in csv format, with the following columns:
  - `d`: Degree of the spline space (from 1 to 16).
  - `k`: Continuity between elements (from 0 to `d-1`).
  - `n_e`: Number of elements in the partition (from 2 to 50).
  - `nepoc`: Number of epochs required for the algorithm to converge to the rule. There is a limit in `10001`, explained in the paper.
  - `x`: Quadrature points. Stored as a string that can be interpreted as a Python list with `eval(x)`. Values are in double precision.
  - `w`: Weight of each point. Stored in the same way that `x`.
- `nonuniform_rules.csv`: Contains the knot sequences and rules calculated for the experiment with non-uniform partitions (described in Section 4.2 in the paper). The file is also in csv format, with the following columns:
  - `d`: Degree of the spline space.
  - `k`: Continuity between elements.
  - `npart`: Unique number to identify the random partition for each specific combination of `d` and `k`.
  - `partition`: Knot sequence determining the partition (excluding the extrema 0 and 1). Stored as a string that can be interpreted as a Python list with `eval(x)`. Values are in double precision.
  - `nepoc`: Number of epochs required for the algorithm to converge to the rule. There is a limit in `10001`, explained in the paper.
  - `x`: Quadrature points. Stored as a string that can be interpreted as a Python list with `eval(x)`. Values are in double precision.
  - `w`: Weight of each point. Stored in the same way that `x`.
  - `loss`: Loss value obtained for the quadrature rule. The rule is considered to be optimal if `loss < 1e-20`.


## License
This software is licensed under the GPL-v3 license.
