import numpy as np
from PIL import Image
import cv2
import os


def combined_display(image, matte):
  # calculate display resolution
  w, h = image.width, image.height
  rw, rh = 800, int(h * 800 / (3 * w))
  
  # obtain predicted foreground
  image = np.asarray(image)
  if len(image.shape) == 2:
    image = image[:, :, None]
  if image.shape[2] == 1:
    image = np.repeat(image, 3, axis=2)
  elif image.shape[2] == 4:
    image = image[:, :, 0:3]
  matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
  foreground = image * matte + np.full(image.shape, 255) * (1 - matte)
  # combine image, foreground, and alpha into one line
  combined = Image.fromarray(np.uint8(foreground))
  return combined

# visualize all images
output_folder = r'demo\image_matting\colab\output'
input_folder=r'demo\image_matting\colab\input'
image_names = os.listdir(input_folder)
for image_name in image_names:
  matte_name = image_name.split('.')[0] + '.png'
  image = Image.open(os.path.join(input_folder, image_name))
  matte = Image.open(os.path.join(output_folder, matte_name))
#   Image.display(combined_display(image, matte))
  image=combined_display(image, matte)
  image.save('out.png')
  print(image_name, '\n')
