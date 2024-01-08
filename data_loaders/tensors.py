import torch

def lengths_to_mask(lengths, max_len):
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    
def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['input'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['input'][0][0]) for b in notnone_batches]
        
    databatchTensor = torch.stack(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    input = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}
    
    img = [b['cond']for b in notnone_batches]
    cond['y'].update({'cond': img})

    return input, cond

def CAHE_collate(batch):
    adapted_batch = [{
        'input': torch.tensor(b[0].T).float().unsqueeze(1),
        'lengths': b[1],
        'cond': b[2],
    } for b in batch]
    return collate(adapted_batch)