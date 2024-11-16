from PIL import Image
from glob import glob
import os
from tqdm import tqdm


image_paths = glob("datasets/yes-to-the-dress-jpeg/*.jpg")
for image_path in tqdm(image_paths):
    #print(image_path)
    img = Image.open(image_path)
    img = img.crop((0, 0, 512, 512))
    dst_path = os.path.join("datasets/yes-to-the-dress", os.path.basename(image_path))
    img.save(dst_path)
