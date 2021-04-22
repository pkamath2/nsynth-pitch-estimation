import torch.nn as nn
import torch.nn.functional as F

import util

class PitchGRU(nn.Module):
    def __init__(self):
        super().__init__()

        config = util.get_config()
        self.num_features = int(config['num_features'])
        self.hidden_size = int(config['hidden_layer_size'])
        self.dropout = float(config['dropout'])
        self.num_layers = int(config['num_layers'])
        self.lower_pitch_limit = int(config['lower_pitch_limit'])
        self.upper_pitch_limit = int(config['upper_pitch_limit'])
        self.classes = [x for x in range(self.lower_pitch_limit, self.upper_pitch_limit)]

        self.gru = nn.GRU(self.num_features, self.hidden_size, self.num_layers, dropout=self.dropout, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.hidden_size * 2, len(self.classes))

    def forward(self, x):
        x, h = self.gru(x)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x, h #return hidden for viz
