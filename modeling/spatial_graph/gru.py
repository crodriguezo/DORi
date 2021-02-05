import torch
import torch.nn as nn
from utils.rnns import feed_forward_rnn

class GRU(nn.Module):

    def __init__(self, cfg):
        super(GRU, self).__init__()
        self.input_size   = cfg.SPATIAL_GRAPH.GRU.INPUT_SIZE
        self.num_layers   = cfg.SPATIAL_GRAPH.GRU.NUM_LAYERS
        self.hidden_size  = cfg.SPATIAL_GRAPH.GRU.HIDDEN_SIZE
        self.bias         = cfg.SPATIAL_GRAPH.GRU.BIAS
        self.dropout      = cfg.SPATIAL_GRAPH.GRU.DROPOUT
        self.bidirectional= cfg.SPATIAL_GRAPH.GRU.BIDIRECTIONAL
        self.batch_first  = cfg.SPATIAL_GRAPH.GRU.BATCH_FIRST

        self.lstm = nn.GRU(input_size   = self.input_size,
                            hidden_size  = self.hidden_size,
                            num_layers   = self.num_layers,
                            bias         = self.bias,
                            dropout      = self.dropout,
                            bidirectional= self.bidirectional,
                            batch_first = self.batch_first)

    def forward(self, sequences, lengths):
        if lengths is None:
            raise "ERROR in this tail you need lengths of sequences."
        return feed_forward_rnn(self.lstm,
                                sequences,
                                lengths=lengths)

