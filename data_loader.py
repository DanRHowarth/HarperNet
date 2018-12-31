import torch
from torch.utils.data import Dataset, DataLoader

class HarperNetDataset(Dataset):
    ''' class to process the HarperNet dataset'''
    
    def __init__(self, dataset, transform = None):
        '''
        dataset: assumes this is a list of numpy array, label pairs preprocessed by the image_load function
        transforms: assumes we can pass transforms from the torchvision.transforms.Compose
        '''
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        
        # note the squirrely bit here because of the way I have structured the load_images function
        X, y = self.dataset[index][0], self.dataset[index][2]
        
        # to perform transforms on dataset 
        if self.transform:
            X = self.transform(X)
        
        return X, y
