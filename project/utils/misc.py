from torch import Tensor


def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def normalize_dist(t):
    # Normalize  # PLZ DON'T BLOW MY GRADIENT
    return (t - t.mean()) / (t.std() + 1e-10)


def logit_onehot(logits: Tensor):
    return (logits == logits.max(1, keepdim=True)[0]).float()
