from torch.utils.data import Dataset, DataLoader
import glob
import pandas


data_train_path = '../Data/train'


class GamesDataset(Dataset):

    def __init__(self, data_path: str):
        super(GamesDataset, self).__init__()

        data_files = glob.glob(f'{data_train_path}/**')



        self.data = None
        self.labels = None
