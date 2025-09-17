using BSON
using Flux
using PyCall

"""
    convert_pytorch_to_flux_weights(pytorch_path::String, flux_path::String)

Converts PyTorch model weights to Flux BSON format.
This is a utility script to convert the .pth files to .bson format for Julia.
"""

function convert_pytorch_to_flux_weights(pytorch_path::String, flux_path::String)
    # Load PyTorch
    torch = pyimport("torch")
    
    # Load PyTorch weights
    pytorch_state = torch.load(pytorch_path, map_location="cpu")
    
    # Convert to Flux format
    flux_state = Dict()
    
    # Convert linear layer weights and biases
    for (key, value) in pytorch_state
        if occursin("linear_relu_stack.0", key)  # First linear layer
            if occursin("weight", key)
                flux_state["linear1.weight"] = permutedims(value.numpy(), (2, 1))  # Transpose for Flux
            elseif occursin("bias", key)
                flux_state["linear1.bias"] = value.numpy()
            end
        elseif occursin("linear_relu_stack.2", key)  # Second linear layer
            if occursin("weight", key)
                flux_state["linear2.weight"] = permutedims(value.numpy(), (2, 1))  # Transpose for Flux
            elseif occursin("bias", key)
                flux_state["linear2.bias"] = value.numpy()
            end
        end
    end
    
    # Save as BSON
    BSON.bson(flux_path, flux_state)
    println("Converted weights saved to: $flux_path")
end

"""
    load_converted_weights(weights_path::String)

Loads converted BSON weights into a Flux model.
"""
function load_converted_weights(weights_path::String)
    # Load the converted weights
    weights = BSON.load(weights_path)
    
    # Create model structure
    model = NeuralNetwork()
    
    # Load weights into model
    Flux.loadmodel!(model, weights)
    
    return model
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    pytorch_path = "data/ANNmodel_weights_new.pth"
    flux_path = "data/ANNmodel_weights_new.bson"
    
    try
        convert_pytorch_to_flux_weights(pytorch_path, flux_path)
        println("Weight conversion completed successfully!")
    catch e
        println("Error converting weights: $e")
        println("Make sure PyCall and PyTorch are available in your Python environment.")
    end
end
