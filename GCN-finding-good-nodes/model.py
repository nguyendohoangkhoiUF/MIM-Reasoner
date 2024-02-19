import torch

from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F


class GAT(torch.nn.Module):
    def __init__(self, input_size, hidden_dim1=256, hidden_dim2=256, heads=4):
        super().__init__()
        self.input_size = input_size
        self.gat1 = GATv2Conv(input_size, hidden_dim1, heads=heads)
        self.gat2 = GATv2Conv(hidden_dim1 * heads, hidden_dim2, heads=heads)
        self.gat3 = GATv2Conv(hidden_dim2 * heads, 2, heads=heads)

    def forward(self, x, edge_index):
        x = x.to(torch.float)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = self.gat3(x, edge_index)
        output = F.log_softmax(x, dim=1)

        return output
