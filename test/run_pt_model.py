import torch
from torch import nn
from collections import OrderedDict
import numpy as np

activation_dict = nn.ModuleDict({
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
    "tanh": nn.Tanh(),
    "identity": nn.Identity()
})


class MLP(nn.Module):
    """
    Construct a fully-connected neural network with flexible depth, width and
    activation function choices.
    """
    def __init__(self,
                 dimList,
                 activation_type='relu',
                 out_activation_type='tanh',
                 verbose=False):
        super(MLP, self).__init__()

        # Construct module list: if use `Python List`, the modules are not
        # added to computation graph. Instead, we should use `nn.ModuleList()`.
        self.moduleList = nn.ModuleList()
        numLayer = len(dimList) - 1
        for idx in range(numLayer):
            i_dim = dimList[idx]
            o_dim = dimList[idx + 1]
            linear_layer = nn.Linear(i_dim, o_dim)
            if idx == numLayer - 1:
                module = nn.Sequential(
                    OrderedDict([
                        ('linear_1', linear_layer),
                        ('act_1', activation_dict[out_activation_type]),
                    ]))
            else:
                module = nn.Sequential(
                    OrderedDict([
                        ('linear_1', linear_layer),
                        ('act_1', activation_dict[activation_type]),
                    ]))

            self.moduleList.append(module)
        if verbose:
            print(self.moduleList)

        self.tanh = nn.Tanh()


    def forward(self, x):
        for m in self.moduleList:
            x = m(x)
        return self.tanh(x)


    def load_weights(self, var):
        with torch.no_grad():
            self.moduleList[0][0].weight = torch.nn.Parameter(var['actor.latent_pi.0.weight'])
            self.moduleList[0][0].bias = torch.nn.Parameter(var['actor.latent_pi.0.bias'])
            self.moduleList[1][0].weight = torch.nn.Parameter(var['actor.latent_pi.2.weight'])
            self.moduleList[1][0].bias = torch.nn.Parameter(var['actor.latent_pi.2.bias'])
            self.moduleList[2][0].weight = torch.nn.Parameter(var['actor.latent_pi.4.weight'])
            self.moduleList[2][0].bias = torch.nn.Parameter(var['actor.latent_pi.4.bias'])
            self.moduleList[3][0].weight = torch.nn.Parameter(var['actor.latent_pi.6.weight'])
            self.moduleList[3][0].bias = torch.nn.Parameter(var['actor.latent_pi.6.bias'])
            self.moduleList[4][0].weight = torch.nn.Parameter(var['actor.mu.weight'])
            self.moduleList[4][0].bias = torch.nn.Parameter(var['actor.mu.bias'])


input_size = 15
layer_size_list = [input_size, 256, 256, 128, 128, 4]
mlp = MLP(dimList=layer_size_list,
          activation_type='relu',)


# for name, value in mlp.named_parameters():
#     print(value)

# Load weights
policy_path = '/home/allen/gym-pybullet-drones/logs/no_wind_3/best_model/policy.pth'
# var_path = '/home/allen/gym-pybullet-drones/logs/no_wind_3/best_model/pytorch_variables.pth'
var = torch.load(policy_path)
# for key, value in var.items():
#     if 'actor' in key:
#         print(key)
mlp.load_weight(var)


def normalize_obs(state):
    MAX_XY = 1
    MAX_Z = 2
    MAX_LIN_VEL_XY = 3
    MAX_LIN_VEL_Z = 1
    MAX_PITCH_ROLL = np.pi
    clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
    clipped_pos_z = np.clip(state[2], 0, MAX_Z)
    clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
    clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
    clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)
    normalized_pos_xy = clipped_pos_xy / MAX_XY
    normalized_pos_z = clipped_pos_z / MAX_Z
    normalized_rp = clipped_rp / MAX_PITCH_ROLL
    normalized_y = state[9] / np.pi  # No reason to clip
    normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
    normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_Z
    normalized_ang_vel = state[13:16] / np.linalg.norm(
        state[13:16]) if np.linalg.norm(
            state[13:16]) != 0 else state[13:16]
    obs = torch.from_numpy(np.hstack([
        normalized_pos_xy, 
        normalized_pos_z, 
        normalized_rp,
        normalized_y, 
        normalized_vel_xy, 
        normalized_vel_z,
        normalized_ang_vel
    ]).reshape(12, ))
    return obs 

# Inference
with torch.no_grad():
    action_scaled = mlp.forward(obs)

# Convert to numpy, and reshape to the original action shape
action_scaled = action_scaled.cpu().numpy().flatten()

# Unscale action
low, high = -np.ones((4)), np.ones((4))
action = low + (0.5 * (action_scaled + 1.0) * (high - low))

# Actor(
#   (features_extractor): FlattenExtractor(
#     (flatten): Flatten(start_dim=1, end_dim=-1)
#   )
#   (latent_pi): Sequential(
#     (0): Linear(in_features=15, out_features=256, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=256, out_features=256, bias=True)
#     (3): ReLU()
#     (4): Linear(in_features=256, out_features=128, bias=True)
#     (5): ReLU()
#     (6): Linear(in_features=128, out_features=128, bias=True)
#     (7): ReLU()
#   )
#   (mu): Linear(in_features=128, out_features=4, bias=True)
#   (log_std): Linear(in_features=128, out_features=4, bias=True)
# )

