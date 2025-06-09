from torch.nn import functional as F
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def remove_layer(model, k, verbose=False):
    layers = list(model.children())
    if verbose:
        print("Layers in the model:")
        for i, layer in enumerate(layers):
            print(f"{i}: {layer.__class__.__name__}")
        print(f"Removing last {k} layers")
    new_model = torch.nn.Sequential(*layers[:-k])
    return new_model


def compute_metrics(preds_tensor, labels_tensor):
    preds_np = preds_tensor.cpu().numpy()
    labels_np = labels_tensor.cpu().numpy()
    return {
        "accuracy": accuracy_score(labels_np, preds_np),
        "precision": precision_score(labels_np, preds_np, average='macro', zero_division=0),
        "recall": recall_score(labels_np, preds_np, average='macro', zero_division=0),
        "f1": f1_score(labels_np, preds_np, average='macro', zero_division=0),
    }


def focal_loss(inputs, targets, alpha=None, gamma=2.0, reduction='mean'):
    probs = F.softmax(inputs, dim=1)
    targets = targets.view(-1, 1)
    pt = probs.gather(1, targets).squeeze(1)
    log_pt = torch.log(pt + 1e-8)  # numerical stability
    focal_term = (1 - pt) ** gamma
    if alpha is not None:
        at = alpha.gather(0, targets.squeeze())
        loss = -at * focal_term * log_pt
    else:
        loss=-focal_term*log_pt
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
def fast_focal_loss(inputs, targets, alpha=None, gamma=2.0, reduction='mean'):
    ce_loss = F.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_term = (1.0 - pt).pow(gamma)
    loss = focal_term * ce_loss
    alpha_t = alpha.gather(0, targets) # alpha[targets]
    loss = alpha_t * loss
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Invalid reduction type: {reduction}. Choose 'none', 'mean', or 'sum'.")