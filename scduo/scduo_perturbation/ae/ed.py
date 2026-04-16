import torch
import torch.nn as nn

class NetBlock(nn.Module):
    def __init__(
            self,
            nlayer: int,
            dim_list: list,
            act_list: list,
            dropout_rate: float,
            noise_rate: float
    ):

        super(NetBlock, self).__init__()
        self.nlayer = nlayer
        self.noise_dropout = nn.Dropout(noise_rate)
        self.linear_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.activation_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()

        for i in range(nlayer):

            self.linear_list.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            nn.init.xavier_uniform_(self.linear_list[i].weight)
            self.bn_list.append(nn.BatchNorm1d(dim_list[i + 1]))
            self.activation_list.append(act_list[i])
            if not i == nlayer - 1:
                self.dropout_list.append(nn.Dropout(dropout_rate))

    def forward(self, x):

        x = self.noise_dropout(x)
        for i in range(self.nlayer):
            x = self.linear_list[i](x)
            x = self.bn_list[i](x)
            x = self.activation_list[i](x)
            if not i == self.nlayer - 1:
                """ don't use dropout for output to avoid loss calculate break down """
                x = self.dropout_list[i](x)

        return x


from typing import Callable, List, Optional
import torch
import torch.nn as nn  
class MLP(nn.Module):
    def __init__(
        self,
        dims: List[int],
        norm: bool = False,
        dropout: bool = False,
        dropout_p: float = 0.0,
        activation: Callable = nn.ELU,
        final_activation: Optional[str] = None,
        norm_type: str = "batchnorm",
        out_mult: int = 1,                 # ⭐ 新增：输出维度乘数（1=默认；2=encoder）
    ):
        super(MLP, self).__init__()

        # Attributes 
        self.dims = dims
        self.norm = norm
        self.activation = activation

        # MLP 
        layers = []
        for i in range(len(self.dims[:-2])):
            block = []
            block.append(torch.nn.Linear(self.dims[i], self.dims[i+1]))
            if norm: 
                if norm_type == 'batchnorm':
                    block.append(torch.nn.BatchNorm1d(self.dims[i+1]))
                else:
                    block.append(torch.nn.LayerNorm(self.dims[i+1]))
            block.append(self.activation())
            if dropout:
                block.append(torch.nn.Dropout(dropout_p))
            
            layers.append(torch.nn.Sequential(*block))
        
        out_dim = self.dims[-1] * out_mult
        layers.append(nn.Linear(self.dims[-2], out_dim))


        # Compile the neural net
        self.net = torch.nn.Sequential(*layers)
        
        if final_activation == "tanh":
            self.final_activation = torch.nn.Tanh()
        elif final_activation == "sigmoid":
            self.final_activation = torch.nn.Sigmoid()
        elif final_activation is "ReLU":
            self.final_activation = torch.nn.ReLU()
        else:
            self.final_activation = None

    def forward(self, x):

        x = self.net(x)
        if not self.final_activation:
            return x
        else:
            return self.final_activation(x)

