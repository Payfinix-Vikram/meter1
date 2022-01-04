import json
import glob
import argparse
import random
import shutil
import csv
import cv2
import imgaug as ia
import imgaug.augmenters as iaa

from pathlib import Path
from PIL import Image
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

IMG_SIZE=320

def augment_image(image, bb):
  ia.seed(random.randint(0, 999999999))
  seq = iaa.Sequential(
    [
        iaa.Affine(
            scale=(0.6, 1.2),
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        ),  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    ]
  )

  image_aug, bbs_aug = seq(image=image, bounding_boxes=bb)

  is_bb_fully_within_image = True
  for bb_ in bbs_aug.bounding_boxes:
    is_bb_fully_within_image and bb_.is_fully_within_image(image.shape)

  if not is_bb_fully_within_image:
    image_aug, bbs_aug = augment_image(image, bb)

  return image_aug, bbs_aug

def augment(input_path, number):
  for train_gt_path in glob.iglob(input_path + '**/*.txt'):

    if '-aug-' in train_gt_path:
      continue

    print(train_gt_path)
    image_path = train_gt_path.replace('-bbox.txt', '.png')
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    with open(train_gt_path, 'r') as f:
      reader = csv.reader(f)
      bbs = []
      for row in reader:
        bbs.append(BoundingBox(x1=float(row[2])*w, y1=float(row[1])*h, x2=float(row[4])*w, y2=float(row[3])*h))

      bbs_i = BoundingBoxesOnImage(bbs, shape=image.shape,)

      for i in range(int(number)):
        is_bb_fully_within_image = False
        while not is_bb_fully_within_image:
          image_aug, bbs_aug = augment_image(image, bbs_i)
          is_bb_fully_within_image = True
          for bb_ in bbs_aug.bounding_boxes:
            is_bb_fully_within_image = is_bb_fully_within_image and bb_.is_fully_within_image(image.shape)

        image_name = Path(image_path).stem + '-aug-' + str(i) + '.png'
        gt_name = Path(image_path).stem + '-aug-' + str(i) + '-bbox.txt'
        cv2.imwrite(input_path + '/' + image_name, image_aug)
        with open(input_path + '/' + gt_name, 'w') as gt_fp:
          for bb in bbs_aug.bounding_boxes:
            gt_fp.write(f'0,{"{:.3f}".format(bb.y1/h)},{"{:.3f}".format(bb.x1/w)},{"{:.3f}".format(bb.y2/h)},{"{:.3f}".format(bb.x2/w)}') #only one class



def main():
  """
  Helper function to create training/testing data asset files and images
  Reads asset files reccursively form input folder
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", help='Folder where we have stored training images', default='raw_data/training_data/train')
  parser.add_argument("--number", help='Number of augmentations', default='2')
  args = parser.parse_args()
  augment(args.input, args.number)

if __name__=="__main__":
  main()