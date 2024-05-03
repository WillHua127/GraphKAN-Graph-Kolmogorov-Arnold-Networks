import torch
import torch.nn as nn

import numpy as np

class NaiveFourierKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, gridsize=300, addbias=True):
        super(NaiveFourierKANLayer,self).__init__()
        self.gridsize= gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) / 
                                             (np.sqrt(inputdim) * np.sqrt(self.gridsize)))
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self,x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)
        #Starting at 1 because constant terms are in the bias
        k = torch.reshape(torch.arange(1, self.gridsize+1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
        #This should be fused to avoid materializing memory
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)

        
        # #We compute the interpolation of the various functions defined by their fourier coefficient for each input coordinates and we sum them 
        # y =  torch.sum(c * self.fouriercoeffs[0:1], (-2, -1)) 
        # y += torch.sum(s * self.fouriercoeffs[1:2], (-2, -1))
        # if self.addbias:
        #     y += self.bias
        # #End fuse
        
        #You can use einsum instead to reduce memory usage
        #It stills not as good as fully fused but it should help
        #einsum is usually slower though
        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
        y = torch.einsum("dbik,djik->bj", torch.concat([c, s], axis=0), self.fouriercoeffs)
        if self.addbias:
            y += self.bias
        
        y = y.view(outshape)
        return y
        