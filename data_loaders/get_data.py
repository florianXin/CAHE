from torch.utils.data import DataLoader
from data_loaders.tensors import CAHE_collate

def get_dataset_class(name):
    from data_loaders.dataset import CAHEDataset
    return CAHEDataset

def get_collate_fn(name):
    return CAHE_collate

def get_dataset(name, split='train'):
    DATA = get_dataset_class(name)
    dataset = DATA(split=split)
    return dataset

def get_dataset_loader(name, batch_size, split='train'):
    dataset = get_dataset(name, split)
    collate = get_collate_fn(name)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True, collate_fn=collate
    )

    return loader