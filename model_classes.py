import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SimplifiedGCN(torch.nn.Module):
    def __init__(
        self, num_node_features: int, num_classes: int, dropout_rate: float = 0.5
    ) -> None:
        """
        Initializes the SimplifiedGCN model, a two-layer Graph Convolutional Network (GCN).

        Args:
            num_node_features (int): The number of features for each node in the input graph.
            num_classes (int): The number of output classes for the classification task.
            dropout_rate (float, optional): The dropout rate for the dropout layer. Default is 0.5.
        """
        super(SimplifiedGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)
        self.dropout_rate = dropout_rate

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the SimplifiedGCN model.

        Args:
            x (torch.Tensor): Node feature matrix of shape [num_nodes, num_node_features].
            edge_index (torch.Tensor): Graph connectivity matrix in COO format with shape [2, num_edges].

        Returns:
            torch.Tensor: Output predictions of shape [num_nodes, num_classes].
        """
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return x
