"""
This script conatins an implementation of Gaussian-Bernoulli Restricted Boltzmann Machine in PyTorch. It is different from a traditional Boltzmann machine in that the visible layer can take continous values.

The forward function implements the free energy of the RBM. Free Energy of an RBM is essentially the negative log-likelihood of the data plus a constant.

This implementation of GB_RBM can be trained by minimizing the Fisher Divergence between the model's generative probability distribution and the Data's Probability Distribution. As of now, this implementation cannot be used with the much more popular Contrastive Divergence (CD-K) methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GB_RBM(nn.Module):
    """
    Class for defining the Gaussian-Bernoulli Restricted Boltzmann Machine. 

    ...

    Attributes
    ----------
    W : torch.Tensor
        Matrix of weights for edges between the visible nodes and the hidden nodes.
    b : torch.Tensor
        bias term for the visible nodes
    c : torch.Tensor
        bias term for the hidden nodes
    logsigma: torch.Tensor
        log standard deviation associated with the gaussian distributions. Is the same for all visible nodes.
        
    Methods
    -------
    forward(x):
        returns the free energy of each observation in x. 
    """
    def __init__(self, D: int, F: int):
        """
        Parameters
        ----------
        D : int
            Number of Nodes in the visible layer
        F : int
            Number of Nodes in the hidden layer

        Returns
        -------
        None
        """

        super().__init__()
        self.W = nn.Parameter(torch.randn(F, D) * 1e-2)
        self.b = nn.Parameter(torch.zeros(D))
        self.logsigma = nn.Parameter(torch.zeros(1))
        self.c = nn.Parameter(torch.zeros(F))

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Observations from data. Should be of shape (batch_size, D)

        Returns
        -------
        free_energy : torch.Tensor
            Vector of shape (batch_size,) with free energy of each observation
        """
        x2term = -torch.sum((x - self.b.unsqueeze(0)) ** 2, 1) / (
            2 * torch.exp(2 * self.logsigma)
        )
        internal_term = torch.matmul(x, self.W.T) / (torch.exp(2 * self.logsigma))
        internal_term = internal_term + self.c.unsqueeze(0)
        internal_term = F.softplus(internal_term)
        internal_term = internal_term.sum(1)
        free_energy = -internal_term - x2term
        return free_energy
