import sys
import torch
import torch.nn as nn
import numpy as np

# based on: the code of DINO
# https://github.com/facebookresearch/dino/blob/cb711401860da580817918b9167ed73e3eef3dcf/utils.py#L594
class MultiCropWrapper( nn.Module ):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__( self, args, encoder, predictor ):
        super( MultiCropWrapper, self ).__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.args = args


    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]

        # Since x may contain 3D point sets with diffrent number of points,
        # x is split into multiple tensors and each tensor is fed into DNN.
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        all_feats = torch.empty(0).to(x[0].device)

        for end_idx in idx_crops:
            inputs = torch.cat( x[ start_idx : end_idx ] )

            inputs = inputs.permute( 0, 2, 1 )
            feats = self.encoder( inputs )

            # accumulate outputs
            all_feats = torch.cat( ( all_feats, feats ) )

            start_idx = end_idx

        all_preds = self.predictor( all_feats )

        return all_preds, all_feats
