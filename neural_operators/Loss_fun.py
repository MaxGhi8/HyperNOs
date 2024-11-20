"""
This module contains the definition of the loss functions that can be used in the training of the Neural Operator.
"""
import torch
from torch import Tensor
from jaxtyping import Float, Complex, jaxtyped
from beartype import beartype

#########################################
# L^p relative loss for N-D functions
#########################################
class LprelLoss(): 
    """ 
    Sum of relative errors in L^p norm 
    
    x, y: torch.tensor
          x and y are tensors of shape (n_samples, *n, d_u)
          where *n indicates that the spatial dimensions can be arbitrary
    """
    def __init__(self, p:int, size_mean=False):
        self.p = p
        self.size_mean = size_mean

    @jaxtyped(typechecker=beartype)
    def rel(self, x:Float[Tensor, "n_samples *n d_u"], 
                  y:Float[Tensor, "n_samples *n d_u"]) -> Float[Tensor, "*n_samples"]:
        num_examples = x.size(0)
        
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), p=self.p, dim=1)
        y_norms = torch.norm(y.reshape(num_examples, -1), p=self.p, dim=1)
        
        # check division by zero
        if torch.any(y_norms <= 1e-5):
            raise ValueError("Division by zero")
        
        if self.size_mean is True:
            return torch.mean(diff_norms/y_norms)
        elif self.size_mean is False:
            return torch.sum(diff_norms/y_norms) # sum along batchsize
        elif self.size_mean is None:
            return diff_norms/y_norms # no reduction
        else:
            raise ValueError("size_mean must be a boolean or None")
    
    @jaxtyped(typechecker=beartype)
    def __call__(self, x:Float[Tensor, "n_samples *n d_u"], 
                       y:Float[Tensor, "n_samples *n d_u"]) -> Float[Tensor, "*n_samples"]:
        return self.rel(x, y)

#########################################
#  H1 relative loss 1D
#########################################
class H1relLoss_1D():
    """ 
    Relative H^1 = W^{1,2} norm, approximated with the Fourier transform
    """
    def __init__(self, beta:float=1.0, size_mean:bool=False, alpha:float=1.0):
        self.beta = beta
        self.size_mean = size_mean
        self.alpha = alpha

    @jaxtyped(typechecker=beartype)  
    def rel(self, x:Complex[Tensor, "n_samples n_x d_u"], 
                  y:Complex[Tensor, "n_samples n_x d_u"]) -> Float[Tensor, "*n_samples"]:
        num_examples = x.size(0)
        
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), p=2, dim=1)
        y_norms = torch.norm(y.reshape(num_examples, -1), p=2, dim=1)
        
        # check division by zero
        if torch.any(y_norms <= 1e-5):
            raise ValueError("Division by zero")
        
        if self.size_mean is True:
            return torch.mean(diff_norms/y_norms)
        elif self.size_mean is False:
            return torch.sum(diff_norms/y_norms) # sum along batchsize
        elif self.size_mean is None:
            return diff_norms/y_norms # no reduction
        else:
            raise ValueError("size_mean must be a boolean or None")
    
    @jaxtyped(typechecker=beartype)
    def __call__(self, x:Float[Tensor, "n_samples n_y d_u"], 
                       y:Float[Tensor, "n_samples n_y d_u"]) -> Float[Tensor, "*n_samples"]:
        n_x, d_u = x.size()[1:]

        k_x = torch.cat(
            (torch.arange(start=0, end=n_x//2, step=1),
             torch.arange(start=-n_x//2, end=0, step=1)), 0).reshape(n_x, 1).repeat(1, d_u)
        k_x = torch.abs(k_x).reshape(1, n_x, d_u).to(x.device)

        x = torch.fft.fftn(x, dim=[1])
        y = torch.fft.fftn(y, dim=[1])
        
        weight = self.alpha*1 + self.beta*(k_x**2)
        loss = self.rel(x*torch.sqrt(weight), y*torch.sqrt(weight)) # Hadamard multiplication

        return loss

#########################################
#  H1 relative loss for 2D functions
#########################################
class H1relLoss():
    """ 
    Relative H^1 = W^{1,2} norm, approximated with the Fourier transform
    """
    def __init__(self, beta:float=1.0, size_mean:bool=False, alpha:float=1.0):
        self.beta = beta
        self.size_mean = size_mean
        self.alpha = alpha

    @jaxtyped(typechecker=beartype)  
    def rel(self, x:Complex[Tensor, "n_samples n_x n_y d_u"], 
                  y:Complex[Tensor, "n_samples n_x n_y d_u"]) -> Float[Tensor, "*n_samples"]:
        num_examples = x.size(0)
        
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), p=2, dim=1)
        y_norms = torch.norm(y.reshape(num_examples, -1), p=2, dim=1)

        # check division by zero
        if torch.any(y_norms <= 1e-5):
            raise ValueError("Division by zero")
        
        if self.size_mean is True:
            return torch.mean(diff_norms/y_norms)
        elif self.size_mean is False:
            return torch.sum(diff_norms/y_norms) # sum along batchsize
        elif self.size_mean is None:
            return diff_norms/y_norms # no reduction
        else:
            raise ValueError("size_mean must be a boolean or None")
    
    @jaxtyped(typechecker=beartype)
    def __call__(self, x:Float[Tensor, "n_samples n_x n_y d_u"], 
                       y:Float[Tensor, "n_samples n_x n_y d_u"]) -> Float[Tensor, "*n_samples"]:
        n_x, n_y, d_u = x.size()[1:]

        k_x = torch.cat(
            (torch.arange(start=0, end=n_x//2, step=1),
             torch.arange(start=-n_x//2, end=0, step=1)), 0).reshape(n_x, 1, 1).repeat(1, n_y, d_u)
        k_y = torch.cat(
            (torch.arange(start=0, end=n_y//2, step=1),
             torch.arange(start=-n_y//2, end=0, step=1)), 0).reshape(1, n_y, 1).repeat(n_x, 1, d_u)
        k_x = torch.abs(k_x).reshape(1 ,n_x, n_y, d_u).to(x.device)
        k_y = torch.abs(k_y).reshape(1, n_x, n_y, d_u).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])
        
        weight = self.alpha*1 + self.beta*(k_x**2 + k_y**2)
        loss = self.rel(x*torch.sqrt(weight), y*torch.sqrt(weight))

        return loss