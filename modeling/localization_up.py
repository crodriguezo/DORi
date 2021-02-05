import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from modeling.spatial_graph.build import SpatialGraph
from utils import loss as L
from utils.rnns import feed_forward_rnn
import utils.pooling as POOLING

class Localization(nn.Module):
    def __init__(self, cfg):
        super(Localization, self).__init__()
        self.cfg = cfg
        self.batch_size = cfg.BATCH_SIZE_TRAIN

        self.model_spatial = SpatialGraph(cfg)        
        self.multimodal_fc2 = nn.Linear(512, 1)

        self.rnn_localization = nn.GRU(input_size   = cfg.LOCALIZATION.INPUT_SIZE,
                                        hidden_size  = cfg.LOCALIZATION.HIDDEN_SIZE,
                                        num_layers   = cfg.LOCALIZATION.NUM_LAYERS,
                                        bias         = cfg.LOCALIZATION.BIAS,
                                        dropout      = cfg.LOCALIZATION.DROPOUT,
                                        bidirectional= cfg.LOCALIZATION.BIDIRECTIONAL,
                                        batch_first = cfg.LOCALIZATION.BATCH_FIRST)


        self.pooling = POOLING.MeanPoolingLayer()
        self.starting = nn.Linear(cfg.CLASSIFICATION.INPUT_SIZE, cfg.CLASSIFICATION.OUTPUT_SIZE)
        self.ending = nn.Linear(cfg.CLASSIFICATION.INPUT_SIZE, cfg.CLASSIFICATION.OUTPUT_SIZE)

    def attention(self, videoFeat, filter, lengths):
        pred_local = torch.bmm(videoFeat, filter.unsqueeze(2)).squeeze()
        return pred_local

    def get_mask_from_sequence_lengths(self, sequence_lengths: torch.Tensor, max_length: int):
        ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
        range_tensor = ones.cumsum(dim=1)
        return (sequence_lengths.unsqueeze(1) >= range_tensor).long()

    def masked_softmax(self, vector: torch.Tensor, mask: torch.Tensor, dim: int = -1, memory_efficient: bool = False, mask_fill_value: float = -1e32):
        if mask is None:
            result = torch.nn.functional.softmax(vector, dim=dim)
        else:
            mask = mask.float()
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(1)
            if not memory_efficient:
                # To limit numerical errors from large vector elements outside the mask, we zero these out.
                result = torch.nn.functional.softmax(vector * mask, dim=dim)
                result = result * mask
                result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
            else:
                masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
                result = torch.nn.functional.softmax(masked_vector, dim=dim)

        return result + 1e-13

    def mask_softmax(self, feat, mask):
        return self.masked_softmax(feat, mask, memory_efficient=False)

    def kl_div(self, p, gt, length):
        individual_loss = []
        for i in range(length.size(0)):
            vlength = int(length[i])
            ret = gt[i][:vlength] * torch.log(p[i][:vlength]/gt[i][:vlength])
            individual_loss.append(-torch.sum(ret))
        individual_loss = torch.stack(individual_loss)
        return torch.mean(individual_loss), individual_loss

    def forward(self, videoFeat, videoFeat_lengths, objects, objects_lengths, humans, humans_lengths, tokens, tokens_lengths, start, end, localiz):

        newVideoFeat, ObjectFeat, HumanFeat, attentionQHO, attentionQVH, attentionQVO = self.model_spatial(videoFeat, videoFeat_lengths, \
                           objects, objects_lengths, \
                           humans, humans_lengths, \
                           tokens, tokens_lengths)

        mask = self.get_mask_from_sequence_lengths(videoFeat_lengths, int(videoFeat.shape[1]))

        videoFeat = newVideoFeat 
        attention = self.multimodal_fc2(videoFeat).squeeze()
        if videoFeat.shape[0] == 1:
            attention = attention.unsqueeze(0)
        rqrt_length = torch.rsqrt(tokens_lengths.float()).unsqueeze(1).repeat(1, attention.shape[1])
        attention = attention * rqrt_length
        attention = self.mask_softmax(attention, mask)
        
        output, _ = feed_forward_rnn(self.rnn_localization,
                                    videoFeat,
                                    lengths=videoFeat_lengths)


        pred_start = self.starting(output.view(-1, output.size(2))).view(-1,output.size(1),1).squeeze()
        pred_start = self.mask_softmax(pred_start, mask)

        pred_end = self.ending(output.view(-1, output.size(2))).view(-1,output.size(1),1).squeeze()
        pred_end = self.mask_softmax(pred_end, mask)

        start_loss, individual_start_loss = self.kl_div(pred_start, start, videoFeat_lengths)
        end_loss, individual_end_loss     = self.kl_div(pred_end, end, videoFeat_lengths)

        individual_loss = individual_start_loss + individual_end_loss

        atten_loss = torch.sum(-( (1-localiz) * torch.log((1-attention) + 1E-12) ), dim=1) 
        atten_loss = torch.mean(atten_loss)

        if self.cfg.ATTENTION_LOSS:
            total_loss = start_loss + end_loss + atten_loss
        else:
            total_loss = start_loss + end_loss

        return total_loss, individual_loss, pred_start, pred_end, attention, atten_loss, attentionQHO, attentionQVH, attentionQVO
