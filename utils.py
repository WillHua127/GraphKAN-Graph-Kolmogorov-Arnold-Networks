import torch
import numpy as np

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask
    
def random_disassortative_splits(labels, num_classes, trn_percent=0.6, val_percent=0.2):
    # * 0.6 labels for training
    # * 0.2 labels for validation
    # * 0.2 labels for testing
    labels, num_classes = labels.cpu(), num_classes.cpu().numpy()
    indices = []
    for i in range(num_classes):
        index = torch.nonzero((labels == i)).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)
        
    percls_trn = int(round(trn_percent * (labels.size()[0] / num_classes)))
    val_lb = int(round(val_percent * labels.size()[0]))
    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    train_mask = index_to_mask(train_index, size=labels.size()[0])
    val_mask = index_to_mask(rest_index[:val_lb], size=labels.size()[0])
    test_mask = index_to_mask(rest_index[val_lb:], size=labels.size()[0])

    return train_mask, val_mask, test_mask