import os

from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)
        return img, self.label_dir

    def __len__(self):
        return len(self.img_path)


# test ant
ants_dataset = MyDataset(
    root_dir="./hymenoptera_data/train",
    label_dir="ants",
)
# ant = ants_dataset.__getItem__(0)
# image, label = ant

# image.show()
# print(image)
# print(label)

# test bee
bees_dataset = MyDataset(
    root_dir="./hymenoptera_data/train",
    label_dir="bees",
)
# bee = bees_dataset.__getItem__(0)
# image, label = bee

# image.show()
# print(image)
# print(label)

# merge two datasets
train_dataset = ants_dataset + bees_dataset

img, label = train_dataset[200]
img.show()
print(label)
