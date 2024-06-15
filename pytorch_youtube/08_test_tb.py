import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")


# image_path = "./hymenoptera_data/train/ants_image/0013035.jpg"
image_path = "./hymenoptera_data/train/bees_image/1092977343_cb42b38d62.jpg"
img = Image.open(image_path)
img_array = np.array(img)

# print(img_array.shape)

writer.add_image(
    "test",
    img_array,
    global_step=2,
    dataformats="HWC",
)

# e.g., y = x
for i in range(100):
    writer.add_scalar("y=2x", 2 * i, i)

writer.close()

# step1: python3 08_test_tb.py
# step2: tensorboard --logdir=logs --port=6008
# step3: http://localhost:6006/

# lsof -i :6007
# kill -9 PID
