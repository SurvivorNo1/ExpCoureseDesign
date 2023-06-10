from torch.utils.data import Dataset

# 定义数据集
class MnistDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]

        if self.transform:
            data = self.transform(data)
        return data, label