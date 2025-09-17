"""
Julia version of cf_sweep.py - Counterfactual generation with alpha sweep
"""

using PyCall
include("ANNinJulia.jl")

# Load NumPy
np = pyimport("numpy")

function main()
    println("Julia Counterfactual Generation")
    println("=" ^ 40)
    
    # Load real data using PyCall
    println("Loading real data...")
    xpairs = np.load("data/xpairs.npy")
    test_data = np.load("data/test_data_new.npy")
    
    println("xpairs size: $(size(xpairs))")
    println("test_data size: $(size(test_data))")
    
    # Load model
    model = load_trained_model("data/ANNmodel_weights_new.pth")
    
    # Select query (equivalent to test_data[51] in Python)
    query_data = Float64.(test_data[52, :])  # Convert to Float64, Julia is 1-indexed
    
    # Solver setup
    println("Setting up Gurobi solver...")
    
    # Run alpha sweep
    alphas = [1.2, 1.5, 2, 3, 5, 10, 15, 20, 30, 50, 100]
    
    println("Running alpha sweep for CFlabel=0...")
    @time results = run_alpha_sweep(
        model,
        query_data,
        0,  
        alphas,
        xpairs,  
        true  
    )
    
    println("\nResults summary:")
    for alpha in sort(collect(keys(results)))
        result = results[alpha]
        @printf("Alpha %.1f: distance=%.0f, pCF=%.3f\n", 
               alpha, result[:distance_obj], result[:probability_CF])
    end
    
    # println("\nRunning alpha sweep for CFlabel=4...")
    # @time results2 = run_alpha_sweep(
    #     model,
    #     query_data,
    #     4,  # CFlabel
    #     alphas,
    #     true
    # )
    
    println("\nCounterfactual generation completed!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
