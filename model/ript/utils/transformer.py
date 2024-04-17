import torch
from torch import nn, einsum
from einops import repeat, rearrange

# # TODO:
# # 以下のmultihead化されたvector attentionのコードは，head数を2以上にするとやけに遅くなる
# # https://github.com/lucidrains/point-transformer-pytorch/blob/main/point_transformer_pytorch/multihead_point_transformer_pytorch.py
# # 一方で，以下の点群向けのvector attentionのコードではmultiheadは使っていない．
# # https://github.com/POSTECH-CVLab/point-transformer/blob/paconv-codebase/scene_seg/model/point_transformer/point_transformer_modules.py
# # BNやスキップ接続については上記が参考になりそう
#
# 以下の論文ではscalar attentionとvector attentionの比較 (4章)で
# 「scalar attentionがチャンネルごとに違うアテンションを計算できない問題を，マルチヘッドによって緩和している」
# と記述があるので，vector attentionを無理にmultihead化しなくても良さそう．
# https://hszhao.github.io/papers/cvpr20_san.pdf

# based on: https://github.com/lucidrains/point-transformer-pytorch/blob/main/point_transformer_pytorch/point_transformer_pytorch.py
class VectorAttention( nn.Module ) :
    def __init__( self,
                  token_dim,
                  tpr_dim,
                  inner_dim,
                  attn_mlp_hidden_mult = 1,
                  tpr_mlp_hidden_dim = 32,
                  attdrop_type = 0, # 0 means AttentionDrop is not applied
                  attdrop_rate = 0.0 ) :

        super().__init__()

        self.to_qkv = nn.Linear( token_dim, inner_dim * 3, bias = False )

        self.tpr_mlp = nn.Sequential( nn.Linear( tpr_dim, tpr_mlp_hidden_dim ),
                                      nn.BatchNorm1d( tpr_mlp_hidden_dim ),
                                      nn.ReLU(),
                                      nn.Linear( tpr_mlp_hidden_dim, inner_dim )
                                    )

        self.attn_mlp = nn.Sequential( nn.Linear( inner_dim, inner_dim * attn_mlp_hidden_mult ),
                                       nn.ReLU(),
                                       nn.Linear( inner_dim * attn_mlp_hidden_mult, inner_dim )
                                     )
        self.attdrop_type = attdrop_type
        self.attdrop_rate = attdrop_rate

    def forward( self, x, r ) :
        # x: input tokenwise features whose shape is [B, T, C]
        #    where B is batch size, T is number of tokens per data, and C is number of dimensions for each tokenwise feature
        # r: token pair relation features whose shape is [B, T, T, D]
        #    where D is number of dimensions for each token pair relation feature

        B, T, C = x.shape

        # get queries, keys, values
        q, k, v = self.to_qkv( x ).chunk( 3, dim = -1 )

        # use subtraction of queries to keys. i suppose this is a better inductive bias for point clouds than dot product
        qk_rel = q[:, :, None, :] - k[:, None, :, :] # [B, T, T, C]

        # expand values
        v = repeat( v, 'b j d -> b i j d', i = T ) # [B, T, T, C]

        # generate relative embeddings
        rel_emb = torch.reshape( r, [ B*T*T, -1 ] )
        rel_emb = self.tpr_mlp( rel_emb )
        rel_emb = torch.reshape( rel_emb, [ B, T, T, -1 ] )

        # add relative embeddings to value
        v = v + rel_emb

        # # use attention mlp, making sure to add relative embedding first
        sim = self.attn_mlp( qk_rel + rel_emb )

        # attention
        attn = sim.softmax( dim = -2 )

        # DropAttention: A Regularization Method for Fully-Connected Self-Attention Networks
        if( self.training and self.attdrop_type != 0 ) :
            droprate = self.attdrop_rate
            channels = attn.shape[-1]
            if( self.attdrop_type == 1 ) :
                mask = ( torch.cuda.FloatTensor( B, 1, T, 1 ).uniform_() > droprate ).to( torch.float32 ) # sample-level drop attention (same samples are dropped in each row of an attention map)
            elif( self.attdrop_type == 2 ) :
                mask = ( torch.cuda.FloatTensor( B, T, T, 1 ).uniform_() > droprate ).to( torch.float32 ) # sample-level drop attention ((possibly) different samples are dropped in each row of an attention map)
            elif( self.attdrop_type == 3 ) :
                mask = ( torch.cuda.FloatTensor( B, 1, T, channels ).uniform_() > droprate ).to( torch.float32 ) # channel-level drop attention
            elif( self.attdrop_type == 4 ) :
                mask = ( torch.cuda.FloatTensor( B, T, T, channels ).uniform_() > droprate ).to( torch.float32 ) # channel-level drop attention
            else :
                print( "unknown attention drop type is specified: " + str( self.attdrop_type ) )
                quit()
            attn = mask * attn
            attn = attn / ( torch.sum( attn, dim=-2, keepdims=True ) + 1e-6 )

        # aggregate
        # agg = einsum('b i j d, b i j d -> b i d', attn, v) # slow
        agg = torch.sum( attn * v, dim=2 )

        return agg

# based on: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
class ScalarAttention( nn.Module ):
    def __init__( self,
                  token_dim,
                  tpr_dim,
                  inner_dim,
                  attn_mlp_hidden_mult = 2,
                  tpr_mlp_hidden_dim = 32,
                  attdrop_type = 0, # 0 means AttentionDrop is not applied
                  attdrop_rate = 0.0 ) :
        super().__init__()

        self.scale = token_dim ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear( token_dim, inner_dim * 3, bias = False )

        self.tpr_mlp = nn.Sequential( nn.Linear( tpr_dim, tpr_mlp_hidden_dim ),
                                      nn.BatchNorm1d( tpr_mlp_hidden_dim ),
                                      nn.ReLU(),
                                      nn.Linear( tpr_mlp_hidden_dim, token_dim ),
                                      nn.BatchNorm1d( token_dim ),
                                      nn.ReLU()
                                    )

        self.attdrop_type = attdrop_type
        self.attdrop_rate = attdrop_rate

    def forward( self, x, r ) :
        # x: input tokenwise features whose shape is [B, T, C]
        #    where B is batch size, T is number of tokens per data, and C is number of dimensions for each tokenwise feature
        # r: token pair relation features whose shape is [B, T, T, D]
        #    where D is number of dimensions for each token pair relation feature

        B, T, C = x.shape

        # There are multiple possible ways to incoorporate r (token pair relation) into scalar self-attention.
        # Here, r is simply transformed to a tensor having the same size as  x, and is added to x.
        r = torch.reshape( r, [ B*T*T, -1 ] )
        r = self.tpr_mlp( r )
        r = torch.reshape( r, [ B, T, T, -1 ] )
        r, _ = torch.max( r, dim=2 )
        # r = torch.mean( r, dim=2 )
        x = x + r

        # the computation below is scaled dot-product self-attention
        q, k, v = self.to_qkv( x ).chunk( 3, dim = -1 )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend( dots )

        # DropAttention: A Regularization Method for Fully-Connected Self-Attention Networks
        if( self.training and self.attdrop_type != 0 ) :
            droprate = self.attdrop_rate
            if( self.attdrop_type == 1 ) :
                mask = ( torch.cuda.FloatTensor( B, 1, T ).uniform_() > droprate ).to( torch.float32 ) # sample-level drop attention (same samples are dropped in each row of an attention map)
            elif( self.attdrop_type == 2 ) :
                mask = ( torch.cuda.FloatTensor( B, T, T ).uniform_() > droprate ).to( torch.float32 ) # sample-level drop attention ((possibly) different samples are dropped in each row of an attention map)
            else :
                print( "unknown attention drop type is specified: " + str( self.attdrop_type ) )
                quit()
            attn = mask * attn
            attn = attn / ( torch.sum( attn, dim=-1, keepdims=True ) + 1e-6 )

        out = torch.matmul( attn, v )
        return out


class NoAttention( nn.Module ):
    def __init__(self, token_dim,
                       tpr_dim,
                       inner_dim,
                       tpr_mlp_hidden_dim = 16 ) :
        super().__init__()

        self.tpr_mlp = nn.Sequential( nn.Linear( tpr_dim, tpr_mlp_hidden_dim ),
                                      nn.BatchNorm1d( tpr_mlp_hidden_dim ),
                                      nn.ReLU(),
                                      nn.Linear( tpr_mlp_hidden_dim, token_dim ),
                                      nn.BatchNorm1d( token_dim ),
                                      nn.ReLU()
                                    )

        self.out_mlp = nn.Sequential( nn.Linear( token_dim, inner_dim ),
                                        nn.BatchNorm1d( inner_dim ),
                                        nn.ReLU() )

    def forward( self, x, r ):
        # There are multiple possible ways to combine r (token pair relation) with x.
        # Here, tpr is simply transformed to a tensor having the same size as  x, and is added to x.
        B, T, C = x.shape
        r = torch.reshape( r, [ B*T*T, -1 ] )
        r = self.tpr_mlp( r )
        r = torch.reshape( r, [ B, T, T, -1 ] )
        r, _ = torch.max( r, dim=2 )
        # r = torch.mean( r, dim=2 )
        x = x + r

        x = torch.reshape( x, [ B*T, -1 ] )
        x = self.out_mlp( x )
        x = torch.reshape( x, [ B, T, -1 ] )
        return x


class PointTransformer( nn.Module ) :
    def __init__( self, token_dim, # number of dimensions for each input/output tokenwise feature
                        tpr_dim, # number of dimensions for each token pair relation feature
                        attention_type,
                        attdrop_type = 0, # 0 means AttentionDrop is not applied
                        attdrop_rate = 0.0
                 ) :

        super().__init__()

        inner_dim = token_dim // 4
        # inner_dim = token_dim // 8

        if( attention_type == "scalar" ) :
            self.attention = ScalarAttention( token_dim, tpr_dim, inner_dim, attdrop_type=attdrop_type, attdrop_rate=attdrop_rate )
        elif( attention_type == "vector" ) :
            self.attention = VectorAttention( token_dim, tpr_dim, inner_dim, attdrop_type=attdrop_type, attdrop_rate=attdrop_rate )
        elif( attention_type == "no" ) :
            self.attention = NoAttention( token_dim, tpr_dim, inner_dim )
        else :
            print( "unknown attention type is specified: " + attention_type )
            quit()

        self.fc1 = nn.Sequential(
            nn.Linear( token_dim, token_dim ),
            nn.BatchNorm1d( token_dim ) )

        self.fc2 = nn.Sequential(
            nn.Linear( inner_dim, token_dim ),
            nn.BatchNorm1d( token_dim ) )

        self.relu = nn.ReLU()

    def forward( self, x, r ) :
        # x: input tokenwise features whose shape is [B, T, C]
        #    where B is batch size, T is number of tokens per data, and C is number of dimensions for each tokenwise feature
        # r: token pair relation features whose shape is [B, T, T, D]
        #    where D is number of dimensions for each token pair relation feature
        B, T, _ = x.shape
        x = torch.reshape( x, [ B*T, -1 ] )

        y = self.fc1( x )
        skip = y
        y = self.relu( y )

        y = torch.reshape( y, [ B, T, -1 ] )
        y = self.attention( y, r )
        y = torch.reshape( y, [ B*T, -1 ] )

        y = self.fc2( y )
        y = y + skip # both features (y and skip) are batch-normalized and unactivated
        y = self.relu( y )
        out_tokenwise_feat = torch.reshape( y, [ B, T, -1 ] )
        return out_tokenwise_feat
