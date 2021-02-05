import torch
import numpy as np
from utils import rnns

class BatchCollator(object):

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))

        index      = transposed_batch[0]
        videoFeat  = transposed_batch[1]
        objectFeat = transposed_batch[2]
        humanFeat  = transposed_batch[3]
        tokens     = transposed_batch[4]
        start      = transposed_batch[5]
        end        = transposed_batch[6]
        localiz    = transposed_batch[7]
        time_start = transposed_batch[8]
        time_end   = transposed_batch[9]
        factor     = transposed_batch[10]
        fps        = transposed_batch[11]

        objectFeat, objectFeat_lengths = rnns.pad_spatial_sequence(objectFeat)
        humanFeat, humanFeat_lengths = rnns.pad_spatial_sequence(humanFeat)            
        videoFeat, videoFeat_lengths = rnns.pad_sequence(videoFeat)

        localiz, localiz_lengths = rnns.pad_sequence(localiz)
        tokens, tokens_lengths   = rnns.pad_sequence(tokens)
        start, start_lengths = rnns.pad_sequence(start)
        end, end_lengths     = rnns.pad_sequence(end)

        return index, \
               videoFeat, \
               videoFeat_lengths, \
               tokens, \
               tokens_lengths, \
               start,  \
               end, \
               localiz, \
               localiz_lengths, \
               time_start, \
               time_end, \
               factor, \
               fps, \
               objectFeat, \
               objectFeat_lengths, \
               humanFeat, \
               humanFeat_lengths, \

