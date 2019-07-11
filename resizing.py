
"""
Created on Thu Jul 11 01:57:22 2019

@author: Tim

Purpose: Use this to resize your images

"""

size = 128  # Size of images you want
  
from PIL import Image
import os

# Alter the path to your directory where the images are at:
# Have the non-_____your image___ sets and the image you want to classify separated by files

path = "/Users/Tim/Documents/Python/Neural_Network/additional_non_ckt/"
dirs = os.listdir( path )

# Remember to resize for both set of files (your image and non-image)

for item in dirs:
   if os.path.isfile(path+item):
       im = Image.open(path+item)
       imResize = im.resize((size,size), Image.ANTIALIAS)
       output_file_name = os.path.join(path, "resizing_new" + item)
       imResize.save(output_file_name + ' resized.jpg', 'JPEG', quality= 95)

print('All done!')