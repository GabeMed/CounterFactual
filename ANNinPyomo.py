import numpy as np
import tempfile
import torch, torch.onnx
import torch.nn as nn
import pyomo.environ as pe
from omlt import OmltBlock
from omlt.neuralnet import FullSpaceNNFormulation, ReluBigMFormulation
from omlt.io.onnx import write_onnx_model_with_bounds, load_onnx_neural_network_with_bounds

# Model dimensionalities
input_size = 979
hidden_size = 200
output_size = 11


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


def load_trained_model(weights_path: str) -> nn.Module:
    """Load trained PyTorch model"""
    model = NeuralNetwork()
    state = torch.load(weights_path)
    model.load_state_dict(state)
    model.eval()
    return model


def make_input_bounds(num_inputs: int):
    """Create input bounds for binary features"""
    return {i: (0.0, 1.0) for i in range(num_inputs)}


def make_dummy_input(sample_tensor: torch.Tensor, num_inputs: int):
    """Create dummy input for ONNX export"""
    return sample_tensor[0, :-1].view(-1, num_inputs)


def export_and_load_network_definition(nn_model: nn.Module, nn_dummy_input: torch.Tensor, bounds):
    """Export PyTorch model to ONNX and load for OMLT"""
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        torch.onnx.export(
            nn_model,
            nn_dummy_input,
            f,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        )
        write_onnx_model_with_bounds(f.name, None, bounds)
        return load_onnx_neural_network_with_bounds(f.name)


def build_formulation(network_definition, use_milp=True):
    """Build OMLT formulation"""
    if use_milp:
        return ReluBigMFormulation(network_definition)
    else:
        return FullSpaceNNFormulation(network_definition)


def get_model_predicted_label(model, query_data):
    """Get model's predicted label"""
    with torch.no_grad():
        z_query = model.forward(query_data[:-1].view(1, -1))
        probabilities = torch.nn.functional.softmax(z_query, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()
        return predicted_label, probabilities[0].numpy()


def build_counterfactual_model(formulation):
    """Build streamlined counterfactual model"""
    model = pe.ConcreteModel()
    model.nn = OmltBlock()
    model.nn.build_formulation(formulation)
    
    # Binary input variables
    for i in model.nn.inputs_set:
        model.nn.inputs[i].domain = pe.Binary
    
    # Index sets
    model.IDX10 = pe.RangeSet(0, output_size - 1)
    model.IDX978 = pe.RangeSet(0, input_size - 1)
    
    # Factual parameters
    model.factual = pe.Param(model.IDX978, initialize={i: 0 for i in range(input_size)}, mutable=True)
    
    # L0 minimization with explicit flip variables
    model.flip = pe.Var(model.IDX978, domain=pe.Binary)
    model.flip_constraint1 = pe.Constraint(model.IDX978, rule=lambda m, i: m.flip[i] >= m.nn.inputs[i] - m.factual[i])
    model.flip_constraint2 = pe.Constraint(model.IDX978, rule=lambda m, i: m.flip[i] >= m.factual[i] - m.nn.inputs[i])
    model.obj = pe.Objective(expr=sum(model.flip[i] for i in model.IDX978), sense=pe.minimize)
    
    return model


def set_factual_parameters(model, query_data, torch_model):
    """Set factual parameters in model"""
    x_factual = query_data[:-1].numpy()
    predicted_label, _ = get_model_predicted_label(torch_model, query_data)
    
    # Set factual values
    for idx, value in enumerate(x_factual):
        model.factual[idx].value = value
    
    return x_factual, predicted_label


def solve_counterfactual(model, cf_label, f_label, alpha, solver):
    """Solve for counterfactual explanation"""
    # Build other class set (excluding CF and F)
    others = [i for i in model.IDX10 if i not in {cf_label, f_label}]
    model.OTHERS = pe.Set(initialize=others)
    
    # CF must dominate F
    model.CFandF_compete = pe.Constraint(
        rule=lambda m: m.nn.outputs[cf_label] >= m.nn.outputs[f_label] + pe.log(alpha)
    )
    
    # CF must dominate all other classes
    model.CF_dominant = pe.Constraint(
        model.OTHERS, 
        rule=lambda m, i: m.nn.outputs[cf_label] >= m.nn.outputs[i]
    )

    # Solve
    solver.solve(model, tee=False)
    
    # Extract solution
    x_cf = np.array([model.nn.inputs[i].value for i in range(input_size)])
    z_cf = np.array([model.nn.outputs[i].value for i in range(output_size)])
    y_cf = torch.nn.functional.softmax(torch.tensor(z_cf), dim=0).numpy()
    
    return x_cf, y_cf


def analyze_solution(cf_label, f_label, y_cf, x_f, x_cf, distance_obj, xpairs, verbose=False):
    """Analyze counterfactual solution"""
    probability_cf = y_cf[cf_label]
    probability_f = y_cf[f_label] 
    predicted_label = np.argmax(y_cf)
    
    # Find changed features
    changed_mask = ~np.isclose(x_cf, x_f, rtol=1e-10, atol=1e-10)
    changed_positions = np.where(changed_mask)[0]
    changed_genes = [xpairs[idx] for idx in changed_positions]
    
    result = {
        'distance': int(distance_obj),
        'probability_cf': probability_cf,
        'probability_f': probability_f,
        'predicted_label': predicted_label,
        'success': predicted_label == cf_label,
        'changed_positions': changed_positions.tolist(),
        'changed_genes': changed_genes,
        'num_changes': len(changed_positions)
    }
    
    if verbose:
        print(f"CF Label: {cf_label}, Predicted: {predicted_label}")
        print(f"CF Probability: {probability_cf:.3f}, F Probability: {probability_f:.3f}")
        print(f"Changes: {len(changed_positions)}, Success: {result['success']}")
    
    return result


def solve_cf_single(formulation, query_data, cf_label, alpha, solver, xpairs, torch_model, verbose=False):
    """Main function for single counterfactual generation"""
    model = build_counterfactual_model(formulation)
    x_f, f_label = set_factual_parameters(model, query_data, torch_model)
    x_cf, y_cf = solve_counterfactual(model, cf_label, f_label, alpha, solver)
    distance_obj = pe.value(model.obj)
    
    result = analyze_solution(cf_label, f_label, y_cf, x_f, x_cf, distance_obj, xpairs, verbose)
    
    del model  # Clean up
    return result


def solve_cf_alpha_sweep(formulation, query_data, cf_label, alphas, solver, xpairs, torch_model, verbose=False):
    """Run counterfactual generation with alpha sweep"""
    results = {}
    
    for alpha in alphas:
        if verbose:
            print(f"Solving for alpha = {alpha}")
        
        try:
            result = solve_cf_single(formulation, query_data, cf_label, alpha, solver, xpairs, torch_model, verbose)
            
            # Only keep results that show improvement
            if not results or result['probability_cf'] > max(r['probability_cf'] for r in results.values()):
                results[alpha] = result
                
        except Exception as e:
            if verbose:
                print(f"Failed for alpha {alpha}: {e}")
            continue
    
    return results


def run_alpha_sweep(formulation, query_data, CFlabel, alphas, solver, xpairs, torch_model, use_explicit_flips=True):
    """Compatibility wrapper for original API"""
    results = solve_cf_alpha_sweep(formulation, query_data, CFlabel, alphas, solver, xpairs, torch_model, verbose=True)
    
    # Convert to original format
    legacy_results = {
        'xCF': {},
        'yCF': {},
        'pCF': {},
        'pF': {},
        'dist': {},
        'dx_pos': {},
        'dx_pairs': {},
        'dxF': {},
        'dxCF': {},
    }
    
    for alpha, result in results.items():
        legacy_results['pCF'][alpha] = result['probability_cf']
        legacy_results['pF'][alpha] = result['probability_f']
        legacy_results['dist'][alpha] = result['distance']
        legacy_results['dx_pos'][alpha] = result['changed_positions']
        legacy_results['dx_pairs'][alpha] = result['changed_genes']
    
    return legacy_results


def get_cell_types():
    """Cell type mapping"""
    cell_types = {
        0: "B cell",
        1: "epithelial cell", 
        2: "monocyte",
        3: "endothelial cell",
        4: "natural killer cell",
        5: "myeloid cell",
        6: "stromal cell",
        7: "lymphocyte", 
        8: "macrophage",
        9: "granulocyte",
        10: "rand"
    }
    
    for idx, name in cell_types.items():
        print(f"{idx}: {name}")
    
    return cell_types