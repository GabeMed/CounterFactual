import numpy as np
import tempfile
# pytorch for training neural network
import torch, torch.onnx
import torch.nn as nn
# pyomo for optimization
import pyomo.environ as pe
# omlt for interfacing our neural network with pyomo
from omlt import OmltBlock
from omlt.neuralnet import FullSpaceNNFormulation
from omlt.io.onnx import write_onnx_model_with_bounds, load_onnx_neural_network_with_bounds
from omlt.io.onnx import load_onnx_neural_network
import onnx

xpairs = np.load('xpairs.npy')
test_data = np.load('test_data.npy')
test_data = torch.from_numpy(test_data).float()
train_data = np.load('train_data.npy')
train_data = torch.from_numpy(train_data).float()
train_kwargs = {'batch_size': 256}
test_kwargs = {'batch_size': 256}
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

# load weights from trained model
model = NeuralNetwork()
model.load_state_dict(torch.load('ANNmodel_weights.pth'))


input_bounds = {}
for i in range(979):
    input_bounds[i] = (0.0, 1.0)

# define dummy input tensor
dummy_input = test_data[0, :-1].view(-1, 979)


with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
    # export neural network to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        f,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    # write ONNX model and its bounds using OMLT
    write_onnx_model_with_bounds(f.name, None, input_bounds)
    # load the network definition from the ONNX model
    network_definition = load_onnx_neural_network_with_bounds(f.name)
    # IN THIS CASE, WE DON'T HAVE INPUT BOUNDS, SO NOT USE CODES ABOVE
    #onnx_model = onnx.load(f.name)
    #network_definition = load_onnx_neural_network(onnx_model)

for layer_id, layer in enumerate(network_definition.layers):
    print(f"{layer_id}\t{layer}\t{layer.activation}")
formulation = FullSpaceNNFormulation(network_definition)

m = pe.ConcreteModel()
# create an OMLT block for the neural network and build its formulation
m.nn = OmltBlock()
m.nn.build_formulation(formulation)
solver = pe.SolverFactory('gurobi')
# Start Our CF formulation
# set all the x as binary variables since we only have 0-1 in all the dimensions in this case
for i in m.nn.inputs_set:
    m.nn.inputs[i].domain = pe.Binary
# set the output class index, 0 ~ 10
m.IDX10 = pe.RangeSet(0, 10)
# set the query cell index, 0 ~ 978
m.IDX978 = pe.RangeSet(0, 978)
# create a blank (init all zeros) factual parameter
m.factual = pe.Param(m.IDX978, initialize={i: 0 for i in range(979)}, mutable=True)
# objective fn is to minimize the distance between factual and counterfactual
# in this case (all binary variables), we use manhattan distance
# we simply square them since it is equivalent to calculating the absolute value
m.obj = pe.Objective(expr= sum([(m.nn.inputs[i] - m.factual[i])**2 for i in m.IDX978]),
                     sense= pe.minimize)

def generate_factual_param(query_data):
    # create factual of interest as a parameter
    def generate_factual_dic(query_data):
        """
        Args:
            query_data: 1d tensor that represents the factual cell (include the label)

        Returns: factual_dic, a dic {0: 0-1, 1: 0-1, ..., 978: 0-1} that encodes the factual cell's gene expression
                 label, the true cell type
        """
        query_dataArr = query_data.numpy()
        x = query_dataArr[:-1]
        label = query_dataArr[-1]
        print(f'the factual cell type is type {label}, \n'
              f'choose your counterfactual cell type in 0 ~ 10 except {label}.')
        factual_dic = {}
        for i in range(x.size):
            factual_dic[i] = x[i]
        return factual_dic, label
    factual_dic, label = generate_factual_dic(query_data)
    # set values for facutal parameter
    for idx, value in factual_dic.items():
        m.factual[idx].value = value
    xF = query_data[:-1].numpy() # factual parameter in array form
    return xF, label

def search_nearest_CF(CFlabel, alpha):
    # add constraint such that the prediction of the counterfactual is consistent with CFlabel assigned
    # reformulated into inequality constraints that contains a confidence factor, alpha
    def CFconsistency_rule(m, i):
        return m.nn.outputs[CFlabel] >= m.nn.outputs[i] + pe.log(alpha)
    m.CFconsistency = pe.Constraint(m.IDX10 - [CFlabel], rule=CFconsistency_rule)
    solver.solve(m, tee=True)
    # transform the xCF into numpy array
    xCF = np.zeros(979)
    for i in range(979):
        xCF[i] = m.nn.inputs[i].value
    # transform the outputs of final layer into numpy array, into probabilities yCF
    zCF = np.zeros(11)
    for i in range(11):
        zCF[i] = m.nn.outputs[i].value
    zCF_tensor = torch.tensor(zCF)
    # in ANN training, we used CrossEntropyLoss() which contains logsoftmax instead of softmax
    log_probabilities_tensor = torch.nn.functional.log_softmax(zCF_tensor, dim=0)
    yCF = np.exp(log_probabilities_tensor.detach().numpy())
    return xCF, yCF

def solve_CF_one_CFlabel(query_data, CFlabel, alpha):
    xF, label_F = generate_factual_param(query_data)
    xCF, yCF = search_nearest_CF(CFlabel, alpha) # yCF is probabilities of each label
    probability_CF = max(yCF)
    label_CF = np.argmax(yCF)
    print(f'the most possible cell type is {label_CF}, as the same as input CFlabel: {CFlabel}')
    print(f'the most possible cell type is {label_CF}, with probability of : {round(probability_CF*100, 2)}%')
    distance_obj = m.obj()
    dx = xCF != xF
    print(f'nearest distance is {sum(bool(d) for d in dx)}, as the same as obj value {distance_obj}')
    dx_position = [i for i, d in enumerate(dx) if d]
    print(f'the position of change is at {dx_position}')
    dxpair = [xpairs[idx] for idx in dx_position]
    dxF = [int(x) for x in [xF[idx] for idx in dx_position]]
    dxCF = [int(x) for x in [xCF[idx] for idx in dx_position]]
    print(f'dxpair: {dxpair} \n'
          f'dxF   : {dxF} \n'
          f'dCF   : {dxCF}')
    # remember to delete the constraint CFconsistency for the next run
    m.del_component(m.CFconsistency)
    return xCF, yCF, probability_CF, distance_obj, dx, dx_position, dxpair, dxF, dxCF

def solve_CF_all_CFlabels(query_data, alpha):
    XCF = {}
    YCF = {}
    probabilities_CF = {}
    distances = {}
    dxs = {}
    dx_positions ={}
    dxpairs = {}
    dxFs = {}
    dxCFs = {}
    for CFlabel in [i for i in range(11) if i != query_data[-1]]:
        xCF, yCF, probability_CF, distance_obj, dx, dx_position, dxpair, dxF, dxCF = solve_CF_one_CFlabel(query_data, CFlabel, alpha)
        XCF[CFlabel] = xCF
        YCF[CFlabel] = yCF
        probabilities_CF[CFlabel] = probability_CF
        distances[CFlabel] = distance_obj
        dx_positions[CFlabel] = dx_position
        dxpairs[CFlabel] = dxpair
        dxFs[CFlabel] = dxF
        dxCFs[CFlabel] = dxCF
    min_distance_CFlabel = min(distances, key=distances.get)
    nearest_xCF = XCF[min_distance_CFlabel]
    nearest_yCF = YCF[min_distance_CFlabel]
    nearest_probability = probabilities_CF[min_distance_CFlabel]
    nearest_distance_obj = distances[min_distance_CFlabel]
    nearest_dx_position = dx_positions[min_distance_CFlabel]
    nearest_xpair = dxpairs[min_distance_CFlabel]
    nearest_dxF = dxFs[min_distance_CFlabel]
    nearest_dxCF = dxCFs[min_distance_CFlabel]
    print(f'the cell type closest to cell type {query_data[-1]} is cell type {min_distance_CFlabel} \n'
          f'the probability of xCF predicted as cell type {min_distance_CFlabel} is {round(nearest_probability*100, 2)}% \n'
          f'nearest distance is {nearest_distance_obj} \n'
          f'the position of change is at {nearest_dx_position} \n'
          f'nearest dxpair: {nearest_dxpair} \n'
          f'nearest dxF   : {nearest_dxF} \n'
          f'nearest dCF   : {nearest_dxCF}'
          )
    return nearest_xCF, nearest_yCF, nearest_distance_obj, nearest_dx_position, nearest_dxF, nearest_dxCF, min_distance_CFlabel

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

# use the first cell in test data as an example
query_data = test_data[0]
CFlabel = 1.2 # choose the counterfactual cell type of interest here
# set confidence factor alpha (> 1)
alpha = 2
xCF, yCF, probability_CF, distance_obj, dx, dx_position, dxpair, dxF, dxCF = solve_CF_one_CFlabel(query_data, CFlabel, alpha)


# nearest_xCF, nearest_yCF, nearest_distance_obj, nearest_dx_position, nearest_dxF, nearest_dxCF, min_distance_CFlabel = solve_CF_all_CFlabels(query_data, alpha)








