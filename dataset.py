import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class MetaDataset(Dataset):

    def __init__(self, imgs_path, labels, metadata=None, transform=None):

        super().__init__()
        self.imgs_path = imgs_path
        self.labels = labels
        self.metadata = metadata

        # if transform is None, we need to ensure that the PIL image will be transformed to tensor, otherwise we'll get
        # an exception
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, item):

        img = Image.open(self.imgs_path[item]).convert("RGB")

        # Applying the transformations
        img = self.transform(img)

        if self.metadata is None:
            metadata = []
        else:
            metadata = self.metadata[item]

        if self.labels is None:
            labels = []
        else:
            labels = self.labels[item]

        return img, labels, metadata

def get_data_loader(imgs_path, labels, metadata=None, transform=None, batch_size=30, shuf=True, num_workers=4,
                     pin_memory=True):

    dt = MetaDataset(imgs_path, labels, metadata, transform)
    dl = DataLoader(dataset=dt, batch_size=batch_size, shuffle=shuf, num_workers=num_workers,
                          pin_memory=pin_memory)
    return dl
