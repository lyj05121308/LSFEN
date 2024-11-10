from torch.utils.data import Dataset, DataLoader


class selfdataset(Dataset):
    def __init__(self, X, Y1,Y2,Y3):

        self.X, self.Y1,self.Y2,self.Y3 =X, Y1,Y2,Y3

    def __getitem__(self, index):
        return self.X[index], self.Y1[index], self.Y2[index], self.Y3[index]

    def __len__(self):
        return len(self.X)
