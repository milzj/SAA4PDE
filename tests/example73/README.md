The files are taken from

J. Milz: https://github.com/milzj/sNewton4PDEOpt/tree/main/examples/example73, 2021


Changes made in test_example73_convergence_rates.py

- Adapted the function example73 and test_example73_convergence_rates to allow
for the application of the semismooth Newton CG method.
- Used interpolation to approximate desired_state rather than projection.
- Calling prox_box_l1 to compute the prox of the adjoint state.
- Iterating over range(3,p) instead of range(2, p).
