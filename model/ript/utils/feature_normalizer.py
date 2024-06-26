"""
Reference:  Iterative Normalization: Beyond Standardization towards Efficient Whitening, CVPR 2019
- Paper:
- Code: https://github.com/huangleiBuaa/IterNorm
"""
import torch.nn
from torch.nn import Parameter

__all__ = ['iterative_normalization', 'IterNorm']

class iterative_normalization_py(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        X, running_mean, running_wmat, nc, ctx.T, eps, momentum, training = args
        # change NxCxHxW to (G x D) x(NxHxW), i.e., g*d*m
        ctx.g = X.size(1) // nc
        x = X.transpose(0, 1).contiguous().view(ctx.g, nc, -1)
        _, d, m = x.size()
        saved = []
        if training:
            # calculate centered activation by subtracted mini-batch mean
            mean = x.mean(-1, keepdim=True)
            xc = x - mean
            saved.append(xc)
            # calculate covariance matrix
            P = [None] * (ctx.T + 1)
            P[0] = torch.eye(d).to(X).expand(ctx.g, d, d)

            # furuya 20220705
            # Sigma = torch.baddbmm(eps, P[0], 1. / m, xc, xc.transpose(1, 2)) # pytorch 1.0.0
            Sigma = torch.baddbmm(P[0], xc, xc.transpose(1, 2), beta=eps, alpha=1./m )

            # reciprocal of trace of Sigma: shape [g, 1, 1]
            rTr = (Sigma * P[0]).sum((1, 2), keepdim=True).reciprocal_()
            saved.append(rTr)
            Sigma_N = Sigma * rTr
            saved.append(Sigma_N)
            for k in range(ctx.T):
                # furuya 20220705
                # P[k + 1] = torch.baddbmm(1.5, P[k], -0.5, torch.matrix_power(P[k], 3), Sigma_N) # pytorch 1.0.0
                P[k + 1] = torch.baddbmm(P[k], torch.matrix_power(P[k], 3), Sigma_N, beta=1.5, alpha=-0.5 )
            saved.extend(P)
            wm = P[ctx.T].mul_(rTr.sqrt())  # whiten matrix: the matrix inverse of Sigma, i.e., Sigma^{-1/2}

            running_mean.copy_(momentum * mean + (1. - momentum) * running_mean)
            running_wmat.copy_(momentum * wm + (1. - momentum) * running_wmat)

        else:
            xc = x - running_mean
            wm = running_wmat
        xn = wm.matmul(xc)
        Xn = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()
        ctx.save_for_backward(*saved)
        return Xn

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad, = grad_outputs
        saved = ctx.saved_variables
        xc = saved[0]  # centered input
        rTr = saved[1]  # trace of Sigma
        sn = saved[2].transpose(-2, -1)  # normalized Sigma
        P = saved[3:]  # middle result matrix,
        g, d, m = xc.size()

        g_ = grad.transpose(0, 1).contiguous().view_as(xc)
        g_wm = g_.matmul(xc.transpose(-2, -1))
        g_P = g_wm * rTr.sqrt()
        wm = P[ctx.T]
        g_sn = 0
        for k in range(ctx.T, 1, -1):
            P[k - 1].transpose_(-2, -1)
            P2 = P[k - 1].matmul(P[k - 1])
            g_sn += P2.matmul(P[k - 1]).matmul(g_P)
            g_tmp = g_P.matmul(sn)

            # furuya 20220705
            # g_P.baddbmm_(1.5, -0.5, g_tmp, P2) # pytorch 1.0.0
            # g_P.baddbmm_(1, -0.5, P2, g_tmp) # pytorch 1.0.0
            # g_P.baddbmm_(1, -0.5, P[k - 1].matmul(g_tmp), P[k - 1]) # pytorch 1.0.0
            g_P.baddbmm_(g_tmp, P2, beta=1.5, alpha=-0.5)
            g_P.baddbmm_(P2, g_tmp, beta=1, alpha=-0.5)
            g_P.baddbmm_(P[k - 1].matmul(g_tmp), P[k - 1], beta=1, alpha=-0.5)

        g_sn += g_P
        # g_sn = g_sn * rTr.sqrt()
        g_tr = ((-sn.matmul(g_sn) + g_wm.transpose(-2, -1).matmul(wm)) * P[0]).sum((1, 2), keepdim=True) * P[0]
        g_sigma = (g_sn + g_sn.transpose(-2, -1) + 2. * g_tr) * (-0.5 / m * rTr)
        # g_sigma = g_sigma + g_sigma.transpose(-2, -1)

        # furuya 20220705
        # g_x = torch.baddbmm(wm.matmul(g_ - g_.mean(-1, keepdim=True)), g_sigma, xc) # pytorch 1.0.0
        g_x = torch.baddbmm(wm.matmul(g_ - g_.mean(-1, keepdim=True)), g_sigma, xc) # no need to change

        grad_input = g_x.view(grad.size(1), grad.size(0), *grad.size()[2:]).transpose(0, 1).contiguous()
        return grad_input, None, None, None, None, None, None, None

class IterNorm(torch.nn.Module):
    def __init__(self, num_features, num_groups=1, num_channels=None, T=5, dim=4, eps=1e-5, momentum=0.1, affine=True,
                 *args, **kwargs):
        super(IterNorm, self).__init__()
        # assert dim == 4, 'IterNorm is not support 2D'
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.affine = affine
        self.dim = dim
        if num_channels is None:
            num_channels = (num_features - 1) // num_groups + 1
        num_groups = num_features // num_channels
        while num_features % num_channels != 0:
            num_channels //= 2
            num_groups = num_features // num_channels
        assert num_groups > 0 and num_features % num_groups == 0, "num features={}, num groups={}".format(num_features,
            num_groups)
        self.num_groups = num_groups
        self.num_channels = num_channels
        shape = [1] * dim
        shape[1] = self.num_features
        if self.affine:
            self.weight = Parameter(torch.Tensor(*shape))
            self.bias = Parameter(torch.Tensor(*shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_groups, num_channels, 1))
        # running whiten matrix

        # furuya 20220705
        # avoid the runtime error
        # self.register_buffer('running_wm', torch.eye(num_channels).expand(num_groups, num_channels, num_channels))
        self.register_buffer('running_wm', torch.eye(num_channels).expand(num_groups, num_channels, num_channels).clone())

        self.reset_parameters()

    def reset_parameters(self):
        # self.reset_running_stats()
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, X: torch.Tensor):
        X_hat = iterative_normalization_py.apply(X, self.running_mean, self.running_wm, self.num_channels, self.T,
                                                 self.eps, self.momentum, self.training)
        # affine
        if self.affine:
            return X_hat * self.weight + self.bias
        else:
            return X_hat

    def extra_repr(self):
        return '{num_features}, num_channels={num_channels}, T={T}, eps={eps}, ' \
               'momentum={momentum}, affine={affine}'.format(**self.__dict__)

# class DecorrelatedBatchNorm(torch.nn.Module):
#     def __init__(self, num_features, num_groups=32, num_channels=0, dim=4, eps=1e-5, momentum=0.1, affine=True, mode=0,
#                  *args, **kwargs):
#         super(DecorrelatedBatchNorm, self).__init__()
#         if num_channels > 0:
#             num_groups = num_features // num_channels
#         self.num_features = num_features
#         self.num_groups = num_groups
#         assert self.num_features % self.num_groups == 0
#         self.dim = dim
#         self.eps = eps
#         self.momentum = momentum
#         self.affine = affine
#         self.mode = mode
#
#         self.shape = [1] * dim
#         self.shape[1] = num_features
#
#         if self.affine:
#             self.weight = Parameter(torch.Tensor(*self.shape))
#             self.bias = Parameter(torch.Tensor(*self.shape))
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)
#
#         self.register_buffer('running_mean', torch.zeros(num_groups, 1))
#         self.register_buffer('running_projection', torch.eye(num_groups))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         # self.reset_running_stats()
#         if self.affine:
#             nn.init.uniform_(self.weight)
#             nn.init.zeros_(self.bias)
#
#     def forward(self, input: torch.Tensor):
#         size = input.size()
#         assert input.dim() == self.dim and size[1] == self.num_features
#         x = input.view(size[0] * size[1] // self.num_groups, self.num_groups, *size[2:])
#         training = self.mode > 0 or (self.mode == 0 and self.training)
#         x = x.transpose(0, 1).contiguous().view(self.num_groups, -1)
#         if training:
#             mean = x.mean(1, keepdim=True)
#             self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean
#             x_mean = x - mean
#             sigma = x_mean.matmul(x_mean.t()) / x.size(1) + self.eps * torch.eye(self.num_groups, device=input.device)
#             # print('sigma size {}'.format(sigma.size()))
#             u, eig, _ = sigma.svd()
#             scale = eig.rsqrt()
#             wm = u.matmul(scale.diag()).matmul(u.t())
#             self.running_projection = (1. - self.momentum) * self.running_projection + self.momentum * wm
#             y = wm.matmul(x_mean)
#         else:
#             x_mean = x - self.running_mean
#             y = self.running_projection.matmul(x_mean)
#         output = y.view(self.num_groups, size[0] * size[1] // self.num_groups, *size[2:]).transpose(0, 1)
#         output = output.contiguous().view_as(input)
#         if self.affine:
#             output = output * self.weight + self.bias
#         return output
#
#     def extra_repr(self):
#         return '{num_features}, num_groups={num_groups}, eps={eps}, momentum={momentum}, affine={affine}, ' \
#                'mode={mode}'.format(**self.__dict__)

class DecorrelatedBatchNorm(torch.nn.Module):
    def __init__(self, num_features, group_size=32, dim=4, eps=1e-5, momentum=0.1, affine=True,
                 *args, **kwargs):
        super(DecorrelatedBatchNorm, self).__init__()
        self.num_features = num_features
        self.group_size = group_size
        self.num_groups = num_features // group_size
        assert self.num_features % self.num_groups == 0
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.shape = [1] * dim
        self.shape[1] = num_features

        if self.affine:
            self.weight = Parameter(torch.Tensor(*self.shape))
            self.bias = Parameter(torch.Tensor(*self.shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(self.num_groups, self.group_size, 1))
        self.register_buffer('running_projection', torch.eye( self.group_size ).unsqueeze(0).tile( self.num_groups, 1, 1 ))
        self.reset_parameters()

    def reset_parameters(self):
        # self.reset_running_stats()
        if self.affine:
            torch.nn.init.uniform_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor):
        size = input.size()
        B = size[0] # batch size
        C = size[1] # num channels
        assert input.dim() == self.dim and C == self.num_features

        # change BxCxHxW to GxDx(BxHxW) where HxW is optional
        x = input.view( B, self.num_groups, C // self.num_groups, *size[2:] )
        x = x.transpose(0,1)
        x = x.transpose(1,2)
        x = torch.reshape( x, [ self.num_groups, C // self.num_groups, -1 ] )

        if self.training:
            mean = x.mean( dim=2, keepdim=True )
            self.running_mean = ( 1. - self.momentum ) * self.running_mean + self.momentum * mean
            x_mean = x - mean
            sigma = torch.bmm( x_mean, x_mean.transpose(1,2) ) + self.eps * torch.eye( C // self.num_groups, device=input.device ).unsqueeze(0)
            u, eig, _ = sigma.svd()
            scale = eig.rsqrt()
            wm = torch.bmm( torch.bmm( u, torch.diag_embed( scale ) ), u.transpose(1,2) ) # U * scale^(-1/2) * U'
            y = torch.bmm( wm, x_mean )
            self.running_projection = (1. - self.momentum) * self.running_projection + self.momentum * wm
        else:
            x_mean = x - self.running_mean
            y = torch.bmm( self.running_projection, x_mean )

        output = y.view( self.num_groups * C // self.num_groups, B, *size[2:]).transpose(0, 1).contiguous()

        if self.affine:
            output = output * self.weight + self.bias
        return output

    def extra_repr(self):
        return '{num_features}, num_groups={num_groups}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'mode={mode}'.format(**self.__dict__)


if __name__ == '__main__':

    # DBN = DecorrelatedBatchNorm( 64, group_size=16, dim=2 )
    # x = torch.randn( 5, 64 )
    # DBN = DecorrelatedBatchNorm( 8, group_size=4, dim=2 )
    # x = torch.randn( 5, 8 )
    # DBN.train()
    # y = DBN( x )

    import numpy as np
    from matplotlib import pyplot as plt
    x = np.linspace(0.2,1,100)
    y = 0.8*x + np.random.randn(100)*0.1

    fig = plt.figure()
    # plt.scatter(x, y)
    plt.scatter(x[1:-2], y[1:-2])
    plt.scatter(x[0], y[0])
    plt.scatter(x[-1], y[-1])
    fig.savefig("img.png")

    data = np.vstack([x, y]).T

    data = torch.from_numpy( data )
    DBN = DecorrelatedBatchNorm( 2, group_size=2, dim=2, affine=False )
    DBN.train()
    output = DBN( data )
    output = output.to('cpu').detach().numpy().copy()
    print( output.shape )
    # plt.scatter( output[:,0], output[:,1] )
    plt.scatter( output[1:-2,0], output[1:-2,1] )
    plt.scatter( output[0,0], output[0,1] )
    plt.scatter( output[-1,0], output[-1,1] )
    fig.savefig("img2.png")


    quit()
    # ItN = IterNorm(64, num_groups=8, T=5, momentum=1, affine=False)
    # print(ItN)
    # ItN.train()
    # #x = torch.randn(32, 64, 14, 14)
    # x = torch.randn(128, 64)
    # x.requires_grad_()
    # y = ItN(x)
    # z = y.transpose(0, 1).contiguous().view(x.size(1), -1)
    # print(z.matmul(z.t()) / z.size(1))
    #
    # y.sum().backward()
    # print('x grad', x.grad.size())
    #
    # ItN.eval()
    # y = ItN(x)
    # z = y.transpose(0, 1).contiguous().view(x.size(1), -1)
    # print(z.matmul(z.t()) / z.size(1))
