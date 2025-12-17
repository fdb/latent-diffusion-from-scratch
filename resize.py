import os
from PIL import Image
from glob import glob
import os
from tqdm import tqdm

os.makedirs("datasets/yes-to-the-dress-256", exist_ok=True)
image_paths = glob("datasets/yes-to-the-dress/*.png")
for image_path in tqdm(image_paths):
    # print(image_path)
    img = Image.open(image_path)
    img = img.resize((256, 256))
    dst_path = os.path.join(
        "datasets/yes-to-the-dress-256", os.path.basename(image_path)
    )
    img.save(dst_path)
