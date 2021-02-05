import torch
import numpy as np
from torch import nn

import modeling.spatial_graph as SG
import utils.pooling as POOLING

class SpatialGraph(nn.Module):

    def __init__(self, cfg):
        super(SpatialGraph, self).__init__()
        self.cfg = cfg

        self.factor_query = SG.GRU(cfg)

        factory = getattr(POOLING, cfg.SPATIAL_GRAPH.POOLING)
        self.pooling_layer = factory()

        self.factor_obj   = nn.Linear(2048, cfg.SPATIAL_GRAPH.OUTPUT_SIZE, bias=True)
        self.factor_human = nn.Linear(2048, cfg.SPATIAL_GRAPH.OUTPUT_SIZE, bias=True)
        self.factor_video = nn.Linear(1024, cfg.SPATIAL_GRAPH.OUTPUT_SIZE, bias=True)

        self.video_query  = nn.Linear(1024, 512, bias=True) # w1
        self.human_query  = nn.Linear(1024, 512, bias=True) # w2
        self.object_query = nn.Linear(1024, 512, bias=True) # w3

        self.factor_video_human_query  = nn.Linear(1024, 512, bias=True) # w4
        self.factor_video_object_query = nn.Linear(1024, 512, bias=True) # w5
        self.factor_update_video = nn.Linear(512, 512, bias=True)       # w6
        
        # self.factor_object_video_query  = nn.Linear(1024, 512) # w7
        self.factor_object_human_query = nn.Linear(1024, 512, bias=True) # w8
        self.factor_update_object = nn.Linear(512,512, bias=True) # w8
        self.factor_update_human  = nn.Linear(512,512, bias=True) # w9
        # w9
        self.query_head1 = nn.Linear(512,256, bias=True)
        self.query_head2 = nn.Linear(512,256, bias=True)
        self.query_head3 = nn.Linear(512,256, bias=True)

        self.query_value1 = nn.Linear(300,512, bias=True)
        self.query_value2 = nn.Linear(300,512, bias=True)
        self.query_value3 = nn.Linear(300,512, bias=True)

        self.query_key1 = nn.Linear(300,256, bias=True)
        self.query_key2 = nn.Linear(300,256, bias=True)
        self.query_key3 = nn.Linear(300,256, bias=True)

    def attention(self, query, key):
        pred_local = torch.bmm(query, key).squeeze()
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
                masked_vector = vector.masked_fill((1 - mask).bool(), mask_fill_value)
                result = torch.nn.functional.softmax(masked_vector, dim=dim)

        return result + 1e-13

    def mask_softmax(self, feat, mask):
        return self.masked_softmax(feat, mask, memory_efficient=True)

    def forward(self, videoFeat, videoFeat_lengths, objects, objects_lengths, humans, humans_lengths, tokens, tokens_lengths):
        
        output, _  = self.factor_query(tokens, tokens_lengths)        
        nodeQuery  = torch.tanh(self.pooling_layer(output, tokens_lengths))

        q1 = self.query_head1(nodeQuery).unsqueeze(1)
        q2 = self.query_head2(nodeQuery).unsqueeze(1)
        q3 = self.query_head3(nodeQuery).unsqueeze(1)

        k1 = self.query_key1(tokens)
        k2 = self.query_key2(tokens)
        k3 = self.query_key3(tokens)

        nodeQueryHO = self.mask_softmax(self.attention(q1,k1.permute(0,2,1)), self.get_mask_from_sequence_lengths(tokens_lengths, tokens.shape[1]))
        attentionNodeQueryHO = nodeQueryHO
        nodeQueryHO = nodeQueryHO.unsqueeze(2).repeat(1,1,512) * output
        nodeQueryHO = torch.sum(nodeQueryHO, dim=1)
        
        nodeQueryVO = self.mask_softmax(self.attention(q2,k2.permute(0,2,1)), self.get_mask_from_sequence_lengths(tokens_lengths, tokens.shape[1]))
        attentionNodeQueryVO = nodeQueryVO
        nodeQueryVO = nodeQueryVO.unsqueeze(2).repeat(1,1,512) * output
        nodeQueryVO = torch.sum(nodeQueryVO, dim=1)
        
        nodeQueryVH = self.mask_softmax(self.attention(q3,k3.permute(0,2,1)), self.get_mask_from_sequence_lengths(tokens_lengths, tokens.shape[1]))
        attentionNodeQueryVH = nodeQueryVH
        nodeQueryVH = nodeQueryVH.unsqueeze(2).repeat(1,1,512) * output
        nodeQueryVH = torch.sum(nodeQueryVH, dim=1)
        
        nodeVideo  = torch.tanh(self.factor_video(videoFeat))
        nodeObject = torch.tanh(self.factor_human(objects))
        nodeHuman  = torch.tanh(self.factor_obj(humans))
        
        updateNodeVideo = nodeVideo
        updateNodeHuman = nodeHuman
        updateNodeObject = nodeObject
        for i in range(self.cfg.SPATIAL_GRAPH.NUMBER_ITERATIONS):
            
            auxx = torch.cat((nodeQueryVH.unsqueeze(1).repeat(1,nodeVideo.size(1), 1), updateNodeVideo), dim=2)
            VQ   = self.video_query(auxx)
            auxx = torch.cat((nodeQueryVH.unsqueeze(1).unsqueeze(1).repeat(1,nodeHuman.size(1), nodeHuman.size(2), 1), updateNodeHuman), dim=3)
            HQ   = self.human_query(auxx)
            MessageHQV = torch.cat((VQ, torch.sum(HQ, dim=2)), dim=2)

            auxx = torch.cat((nodeQueryVO.unsqueeze(1).repeat(1,nodeVideo.size(1), 1), updateNodeVideo), dim=2)
            VQ   = self.video_query(auxx)
            auxx = torch.cat((nodeQueryVO.unsqueeze(1).unsqueeze(1).repeat(1,nodeObject.size(1), nodeObject.size(2), 1), updateNodeObject), dim=3)
            OQ   = self.object_query(auxx) 
            MessageVQO = torch.cat((VQ, torch.sum(OQ, dim=2)), dim=2)

            updateNodeVideo  = nn.functional.relu(self.factor_update_video(self.factor_video_human_query(MessageHQV) * self.factor_video_object_query(MessageVQO)) * nodeVideo)

            
            auxx = torch.cat((nodeQueryHO.unsqueeze(1).unsqueeze(1).repeat(1,nodeObject.size(1), nodeObject.size(2), 1), updateNodeObject), dim=3)
            OQ   = self.object_query(auxx) 
            auxx = torch.cat((nodeQueryHO.unsqueeze(1).unsqueeze(1).repeat(1,nodeHuman.size(1), nodeHuman.size(2), 1), updateNodeHuman), dim=3)
            HQ   = self.human_query(auxx)
            MessageOQH = torch.cat((OQ, torch.sum(HQ, dim=2).unsqueeze(2).repeat(1,1,nodeObject.size(2),1)),dim=3)
            
            auxx = torch.cat((nodeQueryVO.unsqueeze(1).repeat(1,nodeVideo.size(1), 1), updateNodeVideo), dim=2)
            VQ   = self.video_query(auxx)
            auxx = torch.cat((nodeQueryVO.unsqueeze(1).unsqueeze(1).repeat(1,nodeObject.size(1), nodeObject.size(2), 1), updateNodeObject), dim=3)
            OQ   = self.object_query(auxx) 
            MessageVQO = torch.cat((OQ, VQ.unsqueeze(2).repeat(1,1,nodeObject.size(2),1)), dim=3)

            updateNodeObject = nn.functional.relu(self.factor_update_object(self.factor_object_human_query(MessageOQH) * self.factor_video_object_query(MessageVQO)) * nodeObject)

            auxx = torch.cat((nodeQueryHO.unsqueeze(1).unsqueeze(1).repeat(1,nodeObject.size(1), nodeObject.size(2), 1), updateNodeObject), dim=3)
            OQ   = self.object_query(auxx) 
            auxx = torch.cat((nodeQueryHO.unsqueeze(1).unsqueeze(1).repeat(1,nodeHuman.size(1), nodeHuman.size(2), 1), updateNodeHuman), dim=3)
            HQ   = self.human_query(auxx)
            MessageOQH = torch.cat((HQ, torch.sum(OQ,dim=2).unsqueeze(2).repeat(1,1,nodeHuman.size(2),1)), dim=3)

            auxx = torch.cat((nodeQueryVH.unsqueeze(1).repeat(1,nodeVideo.size(1), 1), updateNodeVideo), dim=2)
            VQ   = self.video_query(auxx)
            auxx = torch.cat((nodeQueryVH.unsqueeze(1).unsqueeze(1).repeat(1,nodeHuman.size(1), nodeHuman.size(2), 1), updateNodeHuman), dim=3)
            HQ   = self.human_query(auxx)
            MessageHQV = torch.cat((HQ, VQ.unsqueeze(2).repeat(1,1,nodeHuman.size(2),1)), dim=3)

            updateNodeHuman  = nn.functional.relu(self.factor_update_human(self.factor_object_human_query(MessageOQH) * self.factor_video_human_query(MessageHQV)) * nodeHuman)

        return updateNodeVideo, updateNodeObject, updateNodeHuman, attentionNodeQueryHO, attentionNodeQueryVH, attentionNodeQueryVO
