from typing import List

import torch 
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import random_split, Dataset, DataLoader


TRANSFORMS_ONLINE_DATASETS = {'MNIST': transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                        ])
                            }


def load_online_dataset(name: str, root: str = './data',
                        transform: transforms.Compose = None, 
                        download: bool = True, 
                        split_val: bool = True) -> List[Dataset]:
    """
    Fetch an online dataset from the Pytroch collection
    """
    ds_transforms = TRANSFORMS_ONLINE_DATASETS[name] if transform is None else transform
    
    try:
        dataset_class = getattr(datasets, name)
        train_ds = dataset_class(root, train=True, transform=ds_transforms, download=download)
        test_ds = dataset_class(root, train=False, transform=ds_transforms, download=download)
        if split_val: 
            # TODO: RETURNS THE SUBSET INSTEAD OF DATASET  
            train_ds, val_ds = random_split(train_ds, [0.8, 0.2])
            return [train_ds, val_ds, test_ds]
        return [train_ds, test_ds]
    except ImportError:
        print(f'Datset {name} has not been found')


def build_dataloader(ds_list: List[Dataset], batch_size: int, shuffle: bool = True) -> List[DataLoader]:
    """
    Biuld a dataloader for provided dataset
    """
    dataloaders = []
    for ds in ds_list:
        dataloaders.append(DataLoader(ds, batch_size, shuffle))
    
    return dataloaders


