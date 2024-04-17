# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Tokenizer import Tokenizer
from TokenSetTransformer import TokenSetTransformer

class model_RIPT( nn.Module ) :
    def __init__( self, args ) :
        super( model_RIPT, self ).__init__()
        self.tokenizer = Tokenizer( args )
        self.transformer = TokenSetTransformer( args )

    def forward( self, input ) :
        input = input.permute( 0, 2, 1 ) # [B, C, P ] -> [B, P, C]
        tokens, centers, lrfs = self.tokenizer( input )
        feats = self.transformer( tokens, centers, lrfs )
        return feats
