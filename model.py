import torch
from torch import nn
from torch.nn import init

class GraphConvolution(nn.Module):

    def __init__(self, window_size, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weights = nn.Parameter(
            torch.Tensor(window_size,in_features, out_features)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.weights)

    def forward(self, adjacency, nodes):
        """
        :param adjacency: FloatTensor (batch_size, window_size, node_num, node_num)
        :param nodes: FloatTensor (batch_size, window_size, node_num, in_features)
        :return output: FloatTensor (batch_size, window_size, node_num, out_features)
        """
        batch_size = adjacency.size(0)
        window_size, in_features, out_features = self.weights.size()
        weights = self.weights.unsqueeze(0).expand(batch_size, window_size, in_features, out_features)
        output = adjacency.matmul(nodes).matmul(weights)
        return output

class Generator(nn.Module):

    def __init__(self, window_size, node_num, in_features, out_features, lstm_features):
        super(Generator, self).__init__()
        self.window_size = window_size
        self.node_num = node_num
        self.in_features = in_features
        self.out_features = out_features
        self.gcn = GraphConvolution(window_size, in_features, out_features)
        self.lstm = nn.LSTM(
            input_size=out_features * node_num,
            hidden_size=lstm_features,
            num_layers=1,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(lstm_features, node_num * node_num),
            nn.Sigmoid()
        )

    def forward(self, in_shots):
        """
        :param in_shots: FloatTensor (batch_size, window_size, node_num, node_num)
        :return out_shot: FloatTensor (batch_size, node_num * node_num)
        """
        batch_size, window_size, node_num = in_shots.size()[0: 3]
        eye = torch.eye(node_num).cuda().unsqueeze(0).unsqueeze(0).expand(batch_size, window_size, node_num, node_num)
        in_shots = in_shots + eye
        diag = in_shots.sum(dim=-1, keepdim=True).pow(-0.5).expand(in_shots.size()) * eye
        adjacency = diag.matmul(in_shots).matmul(diag)
        nodes = torch.rand(batch_size, window_size, node_num, self.in_features).cuda()
        gcn_output = self.gcn(adjacency, nodes)
        gcn_output = gcn_output.view(batch_size, window_size, -1)
        _, (hn, _) = self.lstm(gcn_output)
        output = self.ffn(hn)
        return output

class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.ffn1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid()
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        :param input: FloatTensor (batch_size, input_size)
        :return: FloatTensor (batch_size,)
        """
        output = self.ffn2(self.ffn1(input)).squeeze(-1)
        return output
