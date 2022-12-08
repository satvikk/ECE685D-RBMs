import torch


def sliced_score_matching(energy_net, samples, n_particles=1):
    """
    Calculates loss associated with Fisher Divergence using the Sliced Score Matching Method. Implements Equation (16) and (17) from https://arxiv.org/pdf/2101.03288.pdf

    Uses the sliced score matching method, where the gradient of the distributions is projected onto random vectors and then the norm calculated.

    Parameters
    ----------
    energy_net : torch.nn.Module
        PyTorch Model whose forward function returns the free energy (negative log-likelihood) of each observation
    samples : torch.Tensor
        Observations for whom loss is to be computed. shape is (batch_size, number of visible nodes)
    n_particles : int
        number of projections to be made. Higher n_particles leads to better learning.

    Returns
    -------
    loss : torch.Tensor
        mean loss computed. This is the relevant loss
    loss1 : torch.Tensor
        positive phase of the loss. Only needed for monitoring training
    loss2 : torch.Tensor
        negative phase of the loss. Only needed for monitoring training
    """
    dup_samples = (
        samples.unsqueeze(0)
        .expand(n_particles, *samples.shape)
        .contiguous()
        .view(-1, *samples.shape[1:])
    )
    dup_samples.requires_grad_(True)
    vectors = torch.randn_like(dup_samples)
    vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True)

    logp = -energy_net(dup_samples).sum()
    grad1 = torch.autograd.grad(logp, dup_samples, create_graph=True)[0]
    gradv = torch.sum(grad1 * vectors)
    loss1 = (torch.sum(grad1 * vectors, dim=-1) ** 2) * 0.5
    grad2 = torch.autograd.grad(gradv, dup_samples, create_graph=True)[0]
    loss2 = torch.sum(vectors * grad2, dim=-1)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)
    loss = loss1 + loss2
    return loss.mean(), loss1.mean(), loss2.mean()
