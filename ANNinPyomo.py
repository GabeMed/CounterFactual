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
test_data = np.load('test_data_new.npy')
test_data = torch.from_numpy(test_data).float()
train_data = np.load('train_data_new.npy')
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
model.load_state_dict(torch.load('ANNmodel_weights_new.pth'))


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

# start our CF problem
solver = pe.SolverFactory('gurobi')
def build_basic_model(formulation):
    m = pe.ConcreteModel()
    # create an OMLT block for the neural network and build its formulation
    m.nn = OmltBlock()
    m.nn.build_formulation(formulation)
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
    m.obj = pe.Objective(expr=sum([(m.nn.inputs[i] - m.factual[i]) ** 2 for i in m.IDX978]),
                         sense=pe.minimize)
    print('basic model built')
    return m

def generate_factual_param(m, query_data):
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
        Flabel = query_dataArr[-1]
        print(f'the factual cell type is type {Flabel}, \n'
              f'choose your counterfactual cell type in 0 ~ 10 except {Flabel}.')
        factual_dic = {}
        for i in range(x.size):
            factual_dic[i] = x[i]
        return factual_dic, Flabel
    factual_dic, Flabel = generate_factual_dic(query_data)
    # set values for facutal parameter
    for idx, value in factual_dic.items():
        m.factual[idx].value = value
    xF = query_data[:-1].numpy() # factual parameter in array form
    return xF, int(Flabel)

def search_nearest_CF(m, CFlabel, Flabel, alpha):
    # add constraint that describes the ratio of probabilities of counterfactual and factual
    # the ratio is controlled by a confidence factor, alpha
    # >1 when 'forcing' counterfactual to be CFlabel cell type
    # <1 when 'allowing' counterfactual to be the same as Flabel cell type
    if isinstance(CFlabel, int):
        def CFandF_compete_rule(m):
            return m.nn.outputs[CFlabel] >= m.nn.outputs[Flabel] + pe.log(alpha)

        m.CFandF_compete = pe.Constraint(rule=CFandF_compete_rule)
        # add constraints such that the probabilities of counterfactual and factual must be greater than other cell types
        #"""
        def CF_dominant_rule(m, i):
            return m.nn.outputs[CFlabel] >= m.nn.outputs[i]
        m.CF_dominant = pe.Constraint(m.IDX10 - [CFlabel, Flabel], rule=CF_dominant_rule)

        def F_dominant_rule(m, i):
            return m.nn.outputs[Flabel] >= m.nn.outputs[i]
        m.F_dominant = pe.Constraint(m.IDX10 - [CFlabel, Flabel], rule=F_dominant_rule)
        #"""
    elif isinstance(CFlabel, list):
        if isinstance(alpha, list):
            raise ValueError("alpha must be dic as multiple CF labels are entered")
        # considering multiple CF cells is of interest
        m.alpha = pe.Param(CFlabel, initialize={key: value for key, value in alpha.items() if key != 'between'})
        def CFandF_compete_rule(m, CFlabeli):
            return m.nn.outputs[CFlabeli] >= m.nn.outputs[Flabel] + pe.log(m.alpha[CFlabeli])
        m.CFandF_compete = pe.Constraint(pe.Set(initialize=CFlabel), rule=CFandF_compete_rule)
        def CFandCF_compete_rule(m): # this constraint hasn't been developed to fit the situation where there are more than three CFlabel
            return m.nn.outputs[CFlabel[0]] >= m.nn.outputs[CFlabel[1]] + pe.log(alpha['between'])
        m.CFandCF_compete = pe.Constraint(rule=CFandCF_compete_rule)
        def CF_dominant_rule(m, i, CFlabeli):
            return m.nn.outputs[CFlabeli] >= m.nn.outputs[i]
        m.CF_dominant = pe.Constraint(m.IDX10 - ([x for x in CFlabel] + [Flabel]), pe.Set(initialize=CFlabel), rule=CF_dominant_rule)
        def F_dominant_rule(m, i):
            return m.nn.outputs[Flabel] >= m.nn.outputs[i]
        m.F_dominant = pe.Constraint(m.IDX10 - ([x for x in CFlabel] + [Flabel]), rule=F_dominant_rule)

    else:
        raise ValueError("CFlabel must be integer or list")

    solver.solve(m, tee=False)
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

def solve_CF(formulation, query_data, CFlabel, alpha):
    m = build_basic_model(formulation)
    xF, Flabel = generate_factual_param(m, query_data)
    xCF, yCF = search_nearest_CF(m, CFlabel, Flabel, alpha) # yCF is probabilities of each label
    if isinstance(CFlabel, int):
        probability_CF = yCF[CFlabel]
        probability_F = yCF[Flabel]
        label_CF = np.argmax(yCF)

        if label_CF == CFlabel:
            print(f'the most possible cell type is {label_CF}, as the same as input CFlabel: {CFlabel}\n'
                  f'the most possible cell type is {label_CF}, with probability of : {round(probability_CF * 100, 2)}% \n'
                  f'the second most possible cell type is {Flabel}, with probability of: {round(probability_F * 100, 2)}%')
        else:
            print(f'the most possible cell type is {label_CF}, not the same as input CFlabel: {CFlabel}, \n'
                  f'but the same as input Flabel: {Flabel}\n'
                  f'the most possible cell type is {label_CF}, with probability of : {round(probability_F * 100, 2)}% \n'
                  f'the second most possible cell type is {CFlabel}, with probability of : {round(probability_CF * 100, 2)}%')

    elif isinstance(CFlabel, list):
        probability_CF = [yCF[CFlabeli] for CFlabeli in CFlabel]
        probability_F = yCF[Flabel]
        label_CF = np.argmax(yCF)

        if label_CF in CFlabel:
            print(f'the most possible cell type is {label_CF}, included in input CFlabel: {CFlabel}\n'
                  f'the most possible cell type is {label_CF}, with probability of : {round(max(probability_CF) * 100, 2)}% \n'
                  f'the other cell type in CFlabel is {[x for x in CFlabel if x != label_CF]}, '
                  f'with probability of : {[round(x * 100, 2) for x in probability_CF if x != max(probability_CF)]}% \n'
                  f'the factual cell type is {Flabel}, with probability of: {round(probability_F * 100, 2)}% \n'
                  )
        else:
            print(f'the most possible cell type is {label_CF}, not included in input CFlabel: {CFlabel}, \n'
                  f'but the same as input Flabel: {Flabel}\n'
                  f'the most possible cell type is {label_CF}, with probability of : {round(probability_F * 100, 2)}% \n'
                  f'CF cell types are {CFlabel}, with probabilities of {[round(x * 100, 2) for x in probability_CF]}%'
                  )

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
    # remember to delete the model for the next run
    del m
    return xCF, yCF, probability_CF, probability_F, distance_obj, dx, dx_position, dxpair, dxF, dxCF

def solve_CF_with_diff_alpha(formulation, query_data, CFlabel, alphas):
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
        xCF, yCF, probability_CF, probability_F, distance_obj, dx, dx_position, dxpair, dxF, dxCF = solve_CF(formulation, query_data,
                                                                                              CFlabel, alpha)
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

"""
you can choose your factual and counterfactual here
query_data = test_data[i] # the ith data point in test data
make sure check the factual label by:
print(query_data[-1])
CFlabel = i # i from 0 to 10
"""
# use the 51st cell in test data as an example
# its label is cell type 7, lymphocyte, because we can use it as an example
# to examine the performance on "lymphocyte differentiate to B cell (type 0) or Natural killer cell (type 4)"
query_data = test_data[51]
"""
query_data = query_data.reshape(1, 980)
z_query = model.forward(query_data[:, :-1].detach())
z = np.zeros(11)
for i in range(11):
    z[i] = z_query[0, i]
z_tensor = torch.tensor(z)
log_probabilities_tensor = torch.nn.functional.log_softmax(z_tensor, dim=0)
y = np.exp(log_probabilities_tensor.detach().numpy())
"""

# 1. You can search for trajetories towards type 0 and type 4 separately
CFlabel = 0 # choose the counterfactual cell type of interest here
# set confidence factor alphas
alphas = [0.01, 0.02, 0.04, 0.08,
          0.1, 0.2, 0.3,
          0.6, 0.8,
          1.2, 1.8,
          3, 5, 8, 10,
          15, 20, 30, 40, 50, 60,
          100, 200, 300, 400
          ]
xCF_dic, yCF_dic, probability_CF_dic, probability_F_dic, distance_obj_dic, dx_position_dic, dxpair_dic, dxF_dic, dxCF_dic = solve_CF_with_diff_alpha(formulation, query_data, CFlabel, alphas)
alpha_key = list(xCF_dic.keys())
print(f'use the alpha key to access the attributes of interest:\n'
      f'{alpha_key}')
print(distance_obj_dic)
print(probability_CF_dic)

CFlabel = 4 # choose the counterfactual cell type of interest here
# set confidence factor alphas
alphas = [1e-4,
          0.01, 0.02, 0.04, 0.08,
          0.1, 0.2, 0.3,
          0.6, 0.8,
          1.2, 1.8,
          3, 5, 8, 10,
          15, 20, 30, 40, 50, 60,
          100, 200, 300, 400
          ]
xCF_dic, yCF_dic, probability_CF_dic, probability_F_dic, distance_obj_dic, dx_position_dic, dxpair_dic, dxF_dic, dxCF_dic = solve_CF_with_diff_alpha(formulation, query_data, CFlabel, alphas)
alpha_key = list(xCF_dic.keys())
print(f'use the alpha key to access the attributes of interest:\n'
      f'{alpha_key}')
print(distance_obj_dic)
print(probability_CF_dic)

"""
# 2. You can fine tune the two values in alpha to find the hybrid cell state where type 0 and type 4 are both considerable
#    but it can't find the trajecotry with change of certain value in alpha to determine which state occurs first
CFlabel = [0, 4] # choose the counterfactual cell type of interest here, which type 0 and type 4 in this case
alpha = {0: 1,
         4: 1.7,
         'between': 0.8} # fine tune these three values
alpha_bw = 0.8
xCF, yCF, probability_CF, probability_F, distance_obj, dx, dx_position, dxpair, dxF, dxCF = solve_CF(formulation, query_data,
                                                                                              CFlabel, alpha)
"""




