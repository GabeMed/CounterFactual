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
        logits = self.linear_relu_stack(x)
        return logits


def load_trained_model(weights_path: str) -> nn.Module:
    model = NeuralNetwork()
    state = torch.load(weights_path)
    model.load_state_dict(state)
    model.eval()
    return model


def make_input_bounds(num_inputs: int):
    return {i: (0.0, 1.0) for i in range(num_inputs)}


def make_dummy_input(sample_tensor: torch.Tensor, num_inputs: int):
    """Create a 2D dummy input with shape (1, num_inputs) from a labeled row."""
    return sample_tensor[0, :-1].view(-1, num_inputs)


def export_and_load_network_definition(nn_model: nn.Module, nn_dummy_input: torch.Tensor, bounds):
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
    """Build formulation - use MILP for ReLU with Gurobi, FullSpace for other cases."""
    if use_milp:
        return ReluBigMFormulation(network_definition)
    else:
        return FullSpaceNNFormulation(network_definition)

def get_model_predicted_label(model, query_data):
    """Get the model's predicted label for the factual data."""
    with torch.no_grad():
        z_query = model.forward(query_data[:-1].view(1, -1))
        log_probabilities_tensor = torch.nn.functional.log_softmax(z_query, dim=1)
        y = np.exp(log_probabilities_tensor.detach().numpy())
        Flabel_pred = np.argmax(y[0])
    return Flabel_pred, y[0]

def build_counterfactual_model(formulation, use_explicit_flips=True):
    model = pe.ConcreteModel()
    model.nn = OmltBlock()
    model.nn.build_formulation(formulation)
    for i in model.nn.inputs_set:
        model.nn.inputs[i].domain = pe.Binary
    model.IDX10 = pe.RangeSet(0, output_size - 1)
    model.IDX978 = pe.RangeSet(0, input_size - 1)
    model.factual = pe.Param(model.IDX978, initialize={i: 0 for i in range(input_size)}, mutable=True)
    
    if use_explicit_flips:
        # Add explicit binary flip variables for true L0 minimization
        model.flip = pe.Var(model.IDX978, domain=pe.Binary)
        model.flip_constraint1 = pe.Constraint(model.IDX978, rule=lambda m, i: m.flip[i] >= m.nn.inputs[i] - m.factual[i])
        model.flip_constraint2 = pe.Constraint(model.IDX978, rule=lambda m, i: m.flip[i] >= m.factual[i] - m.nn.inputs[i])
        model.obj = pe.Objective(expr=sum(model.flip[i] for i in model.IDX978), sense=pe.minimize)
    else:
        # Original squared difference objective (only L0 if features are exactly 0/1)
        model.obj = pe.Objective(expr=sum([(model.nn.inputs[i] - model.factual[i]) ** 2 for i in model.IDX978]),
                             sense=pe.minimize)
    print('basic model built')
    return model


def generate_factual_param(model, query_data, torch_model):
    def generate_factual_dic(query_data):
        """
        Args:
            query_data: 1d tensor that represents the factual cell (include the label)

        Returns: factual_dic, a dict {0: 0-1, 1: 0-1, ..., 978: 0-1} that encodes the factual cell's gene expression
                 Flabel_pred, the model's predicted cell type
        """
        query_dataArr = query_data.numpy()
        x = query_dataArr[:-1]
        Flabel_true = query_dataArr[-1]  # True label from dataset
        
        # Get model's predicted label
        Flabel_pred, y_pred = get_model_predicted_label(torch_model, query_data)
        
        print(f'the factual cell type (dataset) is type {Flabel_true}, \n'
              f'the factual cell type (model predicted) is type {Flabel_pred}, \n'
              f'choose your counterfactual cell type in 0 ~ 10 except {Flabel_pred}.')
        
        factual_dic = {}
        for i in range(x.size):
            factual_dic[i] = x[i]
        return factual_dic, Flabel_pred
    factual_dic, Flabel_pred = generate_factual_dic(query_data)
    for idx, value in factual_dic.items():
        model.factual[idx].value = value
    xF = query_data[:-1].numpy() # factual parameter in array form
    return xF, int(Flabel_pred)


def search_nearest_CF(model, CFlabel, Flabel, alpha, solver):
    if isinstance(CFlabel, int):
        # Build concrete set of other classes (excluding CF and F)
        others = [i for i in model.IDX10 if i not in {CFlabel, Flabel}]
        model.OTHERS = pe.Set(initialize=others)
        
        def CFandF_compete_rule(model):
            return model.nn.outputs[CFlabel] >= model.nn.outputs[Flabel] + pe.log(alpha)

        model.CFandF_compete = pe.Constraint(rule=CFandF_compete_rule)
        
        def CF_dominant_rule(model, i):
            return model.nn.outputs[CFlabel] >= model.nn.outputs[i]
        model.CF_dominant = pe.Constraint(model.OTHERS, rule=CF_dominant_rule)

        # Remove F_dominant constraints - they fight the counterfactual change
    elif isinstance(CFlabel, list):
        if isinstance(alpha, list):
            raise ValueError("alpha must be dic as multiple CF labels are entered")
        model.alpha = pe.Param(CFlabel, initialize={key: value for key, value in alpha.items() if key != 'between'})
        
        # Build concrete set of other classes (excluding CF and F)
        others = [i for i in model.IDX10 if i not in set(CFlabel + [Flabel])]
        model.OTHERS = pe.Set(initialize=others)
        
        def CFandF_compete_rule(model, CFlabeli):
            return model.nn.outputs[CFlabeli] >= model.nn.outputs[Flabel] + pe.log(model.alpha[CFlabeli])
        model.CFandF_compete = pe.Constraint(pe.Set(initialize=CFlabel), rule=CFandF_compete_rule)
        def CFandCF_compete_rule(model):
            return model.nn.outputs[CFlabel[0]] >= model.nn.outputs[CFlabel[1]] + pe.log(alpha['between'])
        model.CFandCF_compete = pe.Constraint(rule=CFandCF_compete_rule)
        def CF_dominant_rule(model, i, CFlabeli):
            return model.nn.outputs[CFlabeli] >= model.nn.outputs[i]
        model.CF_dominant = pe.Constraint(model.OTHERS, pe.Set(initialize=CFlabel), rule=CF_dominant_rule)
        
        # Remove F_dominant constraints - they fight the counterfactual change

    else:
        raise ValueError("CFlabel must be integer or list")

    solver.solve(model, tee=False)
    xCF = np.zeros(input_size)
    for i in range(input_size):
        xCF[i] = model.nn.inputs[i].value
    zCF = np.zeros(output_size)
    for i in range(output_size):
        zCF[i] = model.nn.outputs[i].value
    yCF = _to_probabilities(zCF)
    return xCF, yCF


def _to_probabilities(logits_array):
    zCF_tensor = torch.tensor(logits_array)
    log_probabilities_tensor = torch.nn.functional.log_softmax(zCF_tensor, dim=0)
    return np.exp(log_probabilities_tensor.detach().numpy())


def solve_CF(formulation, query_data, CFlabel, alpha, solver, xpairs, torch_model, use_explicit_flips=True):
    model = build_counterfactual_model(formulation, use_explicit_flips=use_explicit_flips)
    xF, Flabel = generate_factual_param(model, query_data, torch_model)
    xCF, yCF = search_nearest_CF(model, CFlabel, Flabel, alpha, solver) # yCF is probabilities of each label
    distance_obj = pe.value(model.obj)
    result = _report_results(CFlabel, Flabel, yCF, xF, xCF, distance_obj, xpairs)
    del model
    return (xCF, yCF, result['probability_CF'], result['probability_F'],
            distance_obj, result['dx_mask'], result['dx_positions'], result['dx_pairs'], result['dxF'], result['dxCF'])


def _report_results(CFlabel, Flabel, yCF, xF, xCF, distance_obj, xpairs):
    """Print outcome summary and return structured info for downstream use."""
    if isinstance(CFlabel, int):
        probability_CF = yCF[CFlabel]
        probability_F = yCF[Flabel]
    else:
        probability_CF = [yCF[CFlabeli] for CFlabeli in CFlabel]
        probability_F = yCF[Flabel]

    label_CF = int(np.argmax(yCF))

    if isinstance(CFlabel, int):
        if label_CF == CFlabel:
            print(f'the most possible cell type is {label_CF}, as the same as input CFlabel: {CFlabel}\n'
                  f'the most possible cell type is {label_CF}, with probability of : {round(probability_CF * 100, 2)}% \n'
                  f'the second most possible cell type is {Flabel}, with probability of: {round(probability_F * 100, 2)}%')
        else:
            print(f'the most possible cell type is {label_CF}, not the same as input CFlabel: {CFlabel}, \n'
                  f'but the same as input Flabel: {Flabel}\n'
                  f'the most possible cell type is {label_CF}, with probability of : {round(probability_F * 100, 2)}% \n'
                  f'the second most possible cell type is {CFlabel}, with probability of : {round(probability_CF * 100, 2)}%')
    else:
        if label_CF in CFlabel:
            print(f'the most possible cell type is {label_CF}, included in input CFlabel: {CFlabel}\n'
                  f'the most possible cell type is {label_CF}, with probability of : {round(max(probability_CF) * 100, 2)}% \n'
                  f'the other cell type in CFlabel is {[x for x in CFlabel if x != label_CF]}, '
                  f'with probability of : {[round(x * 100, 2) for x in probability_CF if x != max(probability_CF)]}% \n'
                  f'the factual cell type is {Flabel}, with probability of: {round(probability_F * 100, 2)}% \n')
        else:
            print(f'the most possible cell type is {label_CF}, not included in input CFlabel: {CFlabel}, \n'
                  f'but the same as input Flabel: {Flabel}\n'
                  f'the most possible cell type is {label_CF}, with probability of : {round(probability_F * 100, 2)}% \n'
                  f'CF cell types are {CFlabel}, with probabilities of {[round(x * 100, 2) for x in probability_CF]}%')

    # Use np.isclose for robust difference counting to avoid float artifacts
    dx_mask = ~np.isclose(xCF, xF, rtol=1e-10, atol=1e-10)
    actual_flips = sum(dx_mask)
    print(f'nearest distance is {actual_flips}, as the same as obj value {distance_obj}')
    dx_positions = [i for i, d in enumerate(dx_mask) if d]
    print(f'the position of change is at {dx_positions}')
    dx_pairs = [xpairs[idx] for idx in dx_positions]
    # Use proper rounding instead of int() casting for float values
    dxF = [round(float(x)) for x in [xF[idx] for idx in dx_positions]]
    dxCF = [round(float(x)) for x in [xCF[idx] for idx in dx_positions]]
    print(f'dxpair: {dx_pairs} \n'
          f'dxF   : {dxF} \n'
          f'dCF   : {dxCF}')

    return {
        'probability_CF': probability_CF,
        'probability_F': probability_F,
        'dx_mask': dx_mask,
        'dx_positions': dx_positions,
        'dx_pairs': dx_pairs,
        'dxF': dxF,
        'dxCF': dxCF,
    }


def solve_CF_with_diff_alpha(formulation, query_data, CFlabel, alphas, solver, xpairs, torch_model, use_explicit_flips=True):
    """Run a sweep over alphas and collect solutions where CF prob increases."""
    xCF_dic = {}
    yCF_dic = {}
    probability_CF_dic = {}
    probability_F_dic = {}
    distance_obj_dic = {}
    dx_position_dic = {}
    dxpair_dic = {}
    dxF_dic = {}
    dxCF_dic = {}
    last_probability_CF = 0
    for alpha in alphas:
        xCF, yCF, probability_CF, probability_F, distance_obj, dx, dx_position, dxpair, dxF, dxCF = solve_CF(formulation, query_data, CFlabel, alpha, solver, xpairs, torch_model, use_explicit_flips)
        if probability_CF - last_probability_CF > 0.01:
            xCF_dic[alpha] = xCF
            yCF_dic[alpha] = yCF
            probability_CF_dic[alpha] = probability_CF
            probability_F_dic[alpha] = probability_F
            distance_obj_dic[alpha] = distance_obj
            dx_position_dic[alpha] = dx_position
            dxpair_dic[alpha] = dxpair
            dxF_dic[alpha] = dxF
            dxCF_dic[alpha] = dxCF
            last_probability_CF = probability_CF
    return xCF_dic, yCF_dic, probability_CF_dic, probability_F_dic, distance_obj_dic, dx_position_dic, dxpair_dic, dxF_dic, dxCF_dic


def run_alpha_sweep(formulation, query_data, CFlabel, alphas, solver, xpairs, torch_model, use_explicit_flips=True):
    xCF_dic, yCF_dic, probability_CF_dic, probability_F_dic, distance_obj_dic, dx_position_dic, dxpair_dic, dxF_dic, dxCF_dic = solve_CF_with_diff_alpha(formulation, query_data, CFlabel, alphas, solver, xpairs, torch_model, use_explicit_flips)
    alpha_key = list(xCF_dic.keys())
    print(f'use the alpha key to access the attributes of interest:\n{alpha_key}')
    print(distance_obj_dic)
    print(probability_CF_dic)
    return {
        'xCF': xCF_dic,
        'yCF': yCF_dic,
        'pCF': probability_CF_dic,
        'pF': probability_F_dic,
        'dist': distance_obj_dic,
        'dx_pos': dx_position_dic,
        'dx_pairs': dxpair_dic,
        'dxF': dxF_dic,
        'dxCF': dxCF_dic,
    }


def get_cell_types():
    print("0: B cell \n"
          "1: epithelial cell \n"
          "2: monocyte \n"
          "3: endothelial cell \n"
          "4: natural killer cell \n"
          "5: myeloid cell \n"
          "6: stromal cell \n"
          "7: lymphocyte \n"
          "8: macrophage \n"
          "9: granulocyte \n"
          "10: rand \n")




