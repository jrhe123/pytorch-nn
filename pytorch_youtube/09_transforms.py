import cv2
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

writer = SummaryWriter("logs")

image_path = "./hymenoptera_data/train/bees_image/1092977343_cb42b38d62.jpg"
img = Image.open(image_path)

# backward hooks
# grad
# device
# data

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
writer.add_image("tensor_img", tensor_img)

# normalize
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
norm_img = trans_norm(tensor_img)
writer.add_image("norm_img", norm_img, 2)

# resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
resize_img = trans_resize(img)
resize_img = tensor_trans(resize_img)
writer.add_image("resize_img", resize_img, 0)

# compose - resize & toTensor
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose(
    [
        trans_resize_2,
        tensor_trans,
    ]
)
resize_img_2 = trans_compose(img)
writer.add_image("resize_img_2", resize_img_2, 1)

# random crop
trans_random_crop = transforms.RandomCrop([333, 333])
trans_compose_2 = transforms.Compose(
    [
        trans_random_crop,
        tensor_trans,
    ]
)
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("img_crop", img_crop, i)


print(tensor_img[0][0][0])
print(norm_img[0][0][0])

writer.close()

# open-cv
# cv_img = cv2.imread(image_path)
