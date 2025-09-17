import numpy as np
import torch
import pyomo.environ as pe

from ANNinPyomo import (
    load_trained_model,
    make_input_bounds,
    make_dummy_input,
    export_and_load_network_definition,
    build_formulation,
    run_alpha_sweep,
)


def main():
    # Data and resources
    xpairs = np.load('data/xpairs.npy')
    test_data = torch.from_numpy(np.load('data/test_data_new.npy')).float()

    # Model and OMLT setup
    model = load_trained_model('data/ANNmodel_weights_new.pth')
    bounds = make_input_bounds(979)
    dummy = make_dummy_input(test_data, 979)
    network_def = export_and_load_network_definition(model, dummy, bounds)
    formulation = build_formulation(network_def, use_milp=True)  # Use MILP formulation for ReLU

    # Solver
    solver = pe.SolverFactory('gurobi')

    # Select a query
    query_data = test_data[51]

    # Sweeps with alpha >= 1 to force counterfactual dominance
    run_alpha_sweep(
        formulation,
        query_data,
        CFlabel=0,
        alphas=[1.2, 1.5, 2, 3, 5, 10, 15, 20, 30, 50, 100],
        solver=solver,
        xpairs=xpairs,
        torch_model=model,
        use_explicit_flips=True,
    )

    # run_alpha_sweep(
    #     formulation,
    #     query_data,
    #     CFlabel=4,
    #     alphas=[1.2, 1.5, 2, 3, 5, 10, 15, 20, 30, 50, 100],
    #     solver=solver,
    #     xpairs=xpairs,
    #     torch_model=model,
    #     use_explicit_flips=True,
    # )


if __name__ == '__main__':
    main()
