using JuMP
using Gurobi
using Flux
using BSON
using LinearAlgebra
using Random
using Printf
using Statistics
using PyCall
using MathOptAI

# Include the weight conversion function
include("convert_weights.jl")

# Model dimensionalities
const INPUT_SIZE = 979
const HIDDEN_SIZE = 200
const OUTPUT_SIZE = 11

"""
    NeuralNetwork()

Creates a Flux neural network equivalent to the PyTorch model:
- Input layer: 979 → 200
- ReLU activation
- Output layer: 200 → 11
"""
struct NeuralNetwork
    linear1::Dense
    linear2::Dense
end

# Flux.@functor NeuralNetwork  # Deprecated in Flux v0.15+

function NeuralNetwork()
    return NeuralNetwork(
        Dense(INPUT_SIZE, HIDDEN_SIZE),
        Dense(HIDDEN_SIZE, OUTPUT_SIZE)
    )
end

function (nn::NeuralNetwork)(x)
    return nn.linear2(relu.(nn.linear1(x)))
end

"""
    load_trained_model(weights_path::String)

Loads a trained PyTorch model weights file converted to BSON format and creates a Flux Chain.
"""
function load_trained_model(weights_path::String)
    # Convert .pth to .bson if needed
    bson_path = replace(weights_path, ".pth" => ".bson")
    
    if !isfile(bson_path)
        println("Converting PyTorch weights to BSON format...")
        # Use the conversion function from convert_weights.jl
        success = convert_pytorch_to_flux_weights(weights_path, bson_path)
        if !success
            error("Failed to convert PyTorch weights")
        end
    end
    
    # Load the converted weights
    weights = BSON.load(bson_path)
    
    # Create Flux Chain model (MathOptAI works best with Flux.Chain)
    model = Chain(
        Dense(INPUT_SIZE, HIDDEN_SIZE, relu),
        Dense(HIDDEN_SIZE, OUTPUT_SIZE)
    )
    
    # Load weights into model (transpose to match Flux convention)
    model[1].weight .= Float32.(transpose(weights["linear1.weight"]))
    model[1].bias .= Float32.(weights["linear1.bias"])
    model[2].weight .= Float32.(transpose(weights["linear2.weight"]))
    model[2].bias .= Float32.(weights["linear2.bias"])
    
    println("Loaded actual PyTorch weights from $bson_path")
    return model
end

"""
    make_input_bounds(num_inputs::Int)

Creates input bounds for optimization (0.0 to 1.0 for binary features).
"""
function make_input_bounds(num_inputs::Int)
    return [(0.0, 1.0) for _ in 1:num_inputs]
end

"""
    get_model_predicted_label(model::Chain, query_data::Vector{Float64})

Gets the model's predicted label for the factual data.
"""
function get_model_predicted_label(model::Chain, query_data::Vector{Float64})
    x = query_data[1:end-1]  # Remove label
    logits = model(x)
    probabilities = softmax(logits)
    predicted_label = argmax(probabilities) - 1  # Convert to 0-based indexing to match Python
    return predicted_label, probabilities
end

"""
    build_counterfactual_model(neural_network::Chain, use_explicit_flips::Bool=true)

Builds the JuMP optimization model for counterfactual generation using MathOptAI.jl.
"""
function build_counterfactual_model(neural_network::Chain, use_explicit_flips::Bool=true)
    model = Model(Gurobi.Optimizer)
    set_silent(model)  # Suppress Gurobi output for cleaner logs
    
    # Input variables (binary with bounds for MathOptAI)
    @variable(model, 0 <= x[i=1:INPUT_SIZE] <= 1, Bin)
    
    # Factual input parameter (as variables that will be fixed)
    @variable(model, x_factual[i=1:INPUT_SIZE])
    
    # Convert Flux.Chain to MathOptAI.Pipeline with explicit ReLUBigM
    # This is necessary because Flux.Chain ignores config and uses nonlinear ReLU
    pipeline = convert_flux_to_mathoptai_pipeline(neural_network)
    
    y, formulation = MathOptAI.add_predictor(model, pipeline, x)
    
    # Objective: minimize L0 distance (number of flips)
    if use_explicit_flips
        @variable(model, flip[i=1:INPUT_SIZE], Bin)
        # We'll add the factual constraints later when we have the actual values
        @objective(model, Min, sum(flip[i] for i in 1:INPUT_SIZE))
    else
        # We'll add the factual constraints later when we have the actual values
        @objective(model, Min, sum(x[i]^2 for i in 1:INPUT_SIZE))  # Placeholder
    end
    
    # Store the use_explicit_flips flag and formulation in the model
    model.ext[:use_explicit_flips] = use_explicit_flips
    model.ext[:nn_formulation] = formulation
    
    println("Model built with MathOptAI Pipeline (ReLUBigM)")
    return model, x, y, x_factual
end

"""
    convert_flux_to_mathoptai_pipeline(neural_network::Chain)

Converts a Flux.Chain to MathOptAI.Pipeline with explicit ReLUBigM formulation.
"""
function convert_flux_to_mathoptai_pipeline(neural_network::Chain)
    layers = []
    
    for (i, layer) in enumerate(neural_network)
        if isa(layer, Dense)
            # Extract weights and bias
            W = Float64.(layer.weight)  # Convert to Float64
            b = Float64.(layer.bias)
            
            # Add Affine layer
            push!(layers, MathOptAI.Affine(W, b))
            
            # Check if this layer has relu activation
            if layer.σ == relu
                # Add ReLU with Big-M formulation
                push!(layers, MathOptAI.ReLUBigM(1000.0))
            end
        end
    end
    
    return MathOptAI.Pipeline(layers)
end

# Neural network constraints are now handled automatically by MathOptAI.jl
# with ReLUBigM formulation for Gurobi compatibility

"""
    generate_factual_param!(model, query_data::Vector{Float64}, torch_model::Chain)

Sets the factual parameters in the model and returns factual data and predicted label.
"""
function generate_factual_param!(model, query_data::Vector{Float64}, torch_model::Chain)
    x_factual = query_data[1:end-1]  # Remove label
    factual_label_true = Int(round(query_data[end]))  # Round to nearest integer
    
    # Get model's predicted label
    predicted_label, probabilities = get_model_predicted_label(torch_model, query_data)
    
    println("Factual cell type (dataset): $factual_label_true")
    println("Factual cell type (model predicted): $predicted_label")
    println("Choose counterfactual cell type in 0-10 except $predicted_label")
    
    # Fix factual parameter values
    for i in 1:INPUT_SIZE
        fix(model[:x_factual][i], x_factual[i])
    end
    
    # Add factual constraints to the model
    # For explicit flips, add the flip constraints
    if haskey(model.ext, :use_explicit_flips) && model.ext[:use_explicit_flips]
        @constraint(model, flip_constraint1[i=1:INPUT_SIZE], model[:flip][i] >= model[:x][i] - model[:x_factual][i])
        @constraint(model, flip_constraint2[i=1:INPUT_SIZE], model[:flip][i] >= model[:x_factual][i] - model[:x][i])
    else
        # For squared difference objective, set the actual objective
        @objective(model, Min, sum((model[:x][i] - model[:x_factual][i])^2 for i in 1:INPUT_SIZE))
    end
    
    return x_factual, predicted_label
end

"""
    search_nearest_CF(model, CFlabel, Flabel, alpha, solver, y)

Solves for the nearest counterfactual with given constraints.
"""
function search_nearest_CF(model, CFlabel, Flabel, alpha, solver, y)
    # y contains the output layer variables from MathOptAI
    z2 = y  # MathOptAI returns the output variables directly
    
    if isa(CFlabel, Int)
        # Single counterfactual label (convert to 1-based indexing)
        CFlabel_1based = CFlabel + 1
        Flabel_1based = Flabel + 1
        others = [i for i in 1:OUTPUT_SIZE if i != CFlabel_1based && i != Flabel_1based]
        
        # CF must be more probable than F by factor alpha
        @constraint(model, CFandF_compete, z2[CFlabel_1based] >= z2[Flabel_1based] + log(alpha))
        
        # CF must be more probable than all other classes
        @constraint(model, CF_dominant[i in others], z2[CFlabel_1based] >= z2[i])
        
    elseif isa(CFlabel, Vector{Int})
        # Multiple counterfactual labels (convert to 1-based indexing)
        CFlabel_1based = [c + 1 for c in CFlabel]
        Flabel_1based = Flabel + 1
        others = [i for i in 1:OUTPUT_SIZE if !(i in CFlabel_1based) && i != Flabel_1based]
        
        if isa(alpha, Dict)
            # Each CF must be more probable than F
            @constraint(model, CFandF_compete[i in CFlabel_1based], 
                       z2[i] >= z2[Flabel_1based] + log(alpha[i-1]))  # Convert back to 0-based for alpha dict
            
            # CF labels compete with each other
            if haskey(alpha, :between)
                @constraint(model, CFandCF_compete, 
                           z2[CFlabel_1based[1]] >= z2[CFlabel_1based[2]] + log(alpha[:between]))
            end
            
            # CF labels must dominate others
            @constraint(model, CF_dominant[i in others, j in CFlabel_1based], 
                       z2[j] >= z2[i])
        else
            error("alpha must be Dict when using multiple CF labels")
        end
    else
        error("CFlabel must be Int or Vector{Int}")
    end
    
    # Solve the model
    optimize!(model)
    
    status = termination_status(model)
    if status == MOI.OPTIMAL || status == MOI.TIME_LIMIT
        if status == MOI.TIME_LIMIT
            println("Warning: Optimization hit time limit, using best solution found")
        end
        xCF = [value(model[:x][i]) for i in 1:INPUT_SIZE]
        zCF = [value(z2[i]) for i in 1:OUTPUT_SIZE]
        yCF = softmax(zCF)
        return xCF, yCF
    else
        error("Optimization failed: $(status)")
    end
end

"""
    to_probabilities(logits::Vector{Float64})

Converts logits to probabilities using softmax.
"""
function to_probabilities(logits::Vector{Float64})
    return softmax(logits)
end

"""
    solve_CF(torch_model::Chain, query_data::Vector{Float64}, CFlabel, alpha, xpairs, use_explicit_flips::Bool=true)

Main function to solve counterfactual generation using MathOptAI.jl.
"""
function solve_CF(torch_model::Chain, query_data::Vector{Float64}, CFlabel, alpha, xpairs, use_explicit_flips::Bool=true)
    # Build model with MathOptAI neural network integration
    model, x, y, x_factual = build_counterfactual_model(torch_model, use_explicit_flips)
    
    # Set factual parameters (neural network constraints are handled automatically by MathOptAI)
    xF, Flabel = generate_factual_param!(model, query_data, torch_model)
    
    # Solve for counterfactual
    xCF, yCF = search_nearest_CF(model, CFlabel, Flabel, alpha, model, y)
    
    # Calculate distance
    distance_obj = objective_value(model)
    
    # Report results
    result = report_results(CFlabel, Flabel, yCF, xF, xCF, distance_obj, xpairs)
    
    return (xCF, yCF, result[:probability_CF], result[:probability_F],
            distance_obj, result[:dx_mask], result[:dx_positions], 
            result[:dx_pairs], result[:dxF], result[:dxCF])
end

"""
    report_results(CFlabel, Flabel, yCF, xF, xCF, distance_obj, xpairs)

Reports and analyzes the counterfactual results.
"""
function report_results(CFlabel, Flabel, yCF, xF, xCF, distance_obj, xpairs)
    if isa(CFlabel, Int)
        probability_CF = yCF[CFlabel + 1]  # Convert to 1-based indexing
        probability_F = yCF[Flabel + 1]    # Convert to 1-based indexing
    else
        probability_CF = [yCF[i + 1] for i in CFlabel]  # Convert to 1-based indexing
        probability_F = yCF[Flabel + 1]                 # Convert to 1-based indexing
    end
    
    label_CF = argmax(yCF) - 1  # Convert to 0-based indexing for comparison
    
    if isa(CFlabel, Int)
        if label_CF == CFlabel
            @printf("Most probable cell type: %d (matches CFlabel: %d)\n", label_CF, CFlabel)
            @printf("CF probability: %.2f%%\n", probability_CF * 100)
            @printf("F probability: %.2f%%\n", probability_F * 100)
        else
            @printf("Most probable cell type: %d (matches Flabel: %d, not CFlabel: %d)\n", 
                   label_CF, Flabel, CFlabel)
            @printf("F probability: %.2f%%\n", probability_F * 100)
            @printf("CF probability: %.2f%%\n", probability_CF * 100)
        end
    else
        if label_CF in CFlabel
            @printf("Most probable cell type: %d (in CFlabel: %s)\n", label_CF, CFlabel)
            @printf("Max CF probability: %.2f%%\n", maximum(probability_CF) * 100)
            @printf("F probability: %.2f%%\n", probability_F * 100)
        else
            @printf("Most probable cell type: %d (matches Flabel: %d, not in CFlabel: %s)\n", 
                   label_CF, Flabel, CFlabel)
            @printf("F probability: %.2f%%\n", probability_F * 100)
            @printf("CF probabilities: %s\n", [@sprintf("%.2f%%", p*100) for p in probability_CF])
        end
    end
    
    # Calculate differences
    dx_mask = [abs(xCF[i] - xF[i]) > 1e-10 for i in 1:length(xF)]
    actual_flips = sum(dx_mask)
    @printf("Nearest distance: %d (objective value: %.2f)\n", actual_flips, distance_obj)
    
    dx_positions = findall(dx_mask)
    @printf("Change positions: %s\n", dx_positions)
    
    # Get the gene pairs for changed positions
    dx_pairs = [xpairs[idx] for idx in dx_positions]
    dxF = [round(Int, xF[i]) for i in dx_positions]
    dxCF = [round(Int, xCF[i]) for i in dx_positions]
    @printf("dxpairs: %s\n", dx_pairs)
    @printf("dxF: %s\n", dxF)
    @printf("dxCF: %s\n", dxCF)
    
    return Dict(
        :probability_CF => probability_CF,
        :probability_F => probability_F,
        :dx_mask => dx_mask,
        :dx_positions => dx_positions,
        :dx_pairs => dx_pairs,
        :dxF => dxF,
        :dxCF => dxCF
    )
end

"""
    solve_CF_with_diff_alpha(torch_model, query_data, CFlabel, alphas, xpairs, use_explicit_flips::Bool=true)

Runs counterfactual generation with different alpha values.
"""
function solve_CF_with_diff_alpha(torch_model, query_data, CFlabel, alphas, xpairs, use_explicit_flips::Bool=true)
    results = Dict()
    last_probability_CF = 0.0
    
    for alpha in alphas
        println("Solving with alpha = $alpha")
        xCF, yCF, probability_CF, probability_F, distance_obj, dx, dx_position, dxpair, dxF, dxCF = 
            solve_CF(torch_model, query_data, CFlabel, alpha, xpairs, use_explicit_flips)
        
        if isa(probability_CF, Number)
            prob_diff = probability_CF - last_probability_CF
        else
            prob_diff = maximum(probability_CF) - last_probability_CF
        end
        
        if prob_diff > 0.01
            results[alpha] = Dict(
                :xCF => xCF,
                :yCF => yCF,
                :probability_CF => probability_CF,
                :probability_F => probability_F,
                :distance_obj => distance_obj,
                :dx_position => dx_position,
                :dxpair => dxpair,
                :dxF => dxF,
                :dxCF => dxCF
            )
            last_probability_CF = isa(probability_CF, Number) ? probability_CF : maximum(probability_CF)
        end
    end
    
    return results
end

"""
    run_alpha_sweep(torch_model, query_data, CFlabel, alphas, xpairs, use_explicit_flips::Bool=true)

Main function to run alpha sweep and collect results.
"""
function run_alpha_sweep(torch_model, query_data, CFlabel, alphas, xpairs, use_explicit_flips::Bool=true)
    results = solve_CF_with_diff_alpha(torch_model, query_data, CFlabel, alphas, xpairs, use_explicit_flips)
    
    alpha_keys = collect(keys(results))
    println("Alpha keys: $alpha_keys")
    
    for alpha in alpha_keys
        @printf("Alpha %d: distance=%.2f, pCF=%.3f\n", 
               alpha, results[alpha][:distance_obj], results[alpha][:probability_CF])
    end
    
    return results
end

"""
    get_cell_types()

Prints the cell type mapping.
"""
function get_cell_types()
    println("0: B cell")
    println("1: epithelial cell")
    println("2: monocyte")
    println("3: endothelial cell")
    println("4: natural killer cell")
    println("5: myeloid cell")
    println("6: stromal cell")
    println("7: lymphocyte")
    println("8: macrophage")
    println("9: granulocyte")
    println("10: rand")
end

"""
    benchmark_comparison(python_time, julia_time)

Compares execution times between Python and Julia implementations.
"""
function benchmark_comparison(python_time, julia_time)
    speedup = python_time / julia_time
    @printf("Python time: %.3f seconds\n", python_time)
    @printf("Julia time: %.3f seconds\n", julia_time)
    @printf("Julia speedup: %.2fx\n", speedup)
end

# Example usage
function main()
    println("Julia Counterfactual Generation with JuMP and Flux")
    println("=" ^ 50)
    
    # Load model (in practice, convert PyTorch weights)
    model = load_trained_model("data/ANNmodel_weights_new.pth")
    
    # Load real data
    np = pyimport("numpy")
    xpairs = np.load("data/xpairs.npy")
    test_data = np.load("data/test_data_new.npy")
    
    # Example query data (equivalent to test_data[51] in Python)
    query_data = Float64.(test_data[52, :])  # Julia is 1-indexed
    
    # Run alpha sweep
    alphas = [1.2, 1.5, 2, 3, 5, 10, 15, 20, 30, 50, 100]
    
    println("Running alpha sweep...")
    @time results = run_alpha_sweep(model, query_data, 0, alphas, xpairs, true)  # Exact match with Python
    
    println("Counterfactual generation completed!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
