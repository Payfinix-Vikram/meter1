import json
import glob
import argparse
import random
import shutil
import uuid
from PIL import Image

from pathlib import Path

IMG_SIZE=320

def cleanup(output):
  """
  Delete contents of output directory before starting to write to them
  """
  Path(output + "/train").mkdir(parents=True, exist_ok=True)
  Path(output + "/test").mkdir(parents=True, exist_ok=True)

def parse_data(input, output):
  """
  Iterate through asset files, create bounding boxes and create training and test data in 80/20 ratio
  """

  for assetfile in glob.iglob(input + '/**/*.json', recursive=True):
    print(f"Processing {assetfile}")
    asset = json.load(open (assetfile, "r"))
    imagefile = Path(assetfile).parent / asset["imagePath"]

    img = Image.open(imagefile)
    img = img.resize((IMG_SIZE,IMG_SIZE), Image.ANTIALIAS)
    
    name = str(uuid.uuid4())
    if random.uniform(0, 1) > 0.8:
      gtfile = output  + "/test/" + name + "-bbox.txt"
      outimgfile = output + "/test/" + name + '.png'
    else:
      gtfile = output  + "/train/" + name + "-bbox.txt"
      outimgfile = output + "/train/" + name + '.png'

    img.save(outimgfile, "PNG")

    h = asset["imageHeight"]
    w = asset["imageWidth"]

    with open(gtfile, 'w') as gt_fp:
      for shape in asset["shapes"]:
        # bboxes are of format [ymin, xmin], [ymax, xmax]
        xmin = shape["points"][0][0] / w
        ymin = shape["points"][0][1] / h
        xmax = shape["points"][1][0] / w
        ymax = shape["points"][1][1] / h

        gt_fp.write(f'0,{"{:.3f}".format(ymin)},{"{:.3f}".format(xmin)},{"{:.3f}".format(ymax)},{"{:.3f}".format(xmax)}') #only one class


def main():
  """
  Helper function to create training/testing data asset files and images
  Reads asset files reccursively form input folder
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", help='Folder where we have stored training images', default='raw_data/MeterImage')
  parser.add_argument("--output", help='Folder where we store ground truth', default='raw_data/training_data')
  args = parser.parse_args()
  cleanup(args.output)
  parse_data(args.input, args.output)

if __name__=="__main__":
    main()



