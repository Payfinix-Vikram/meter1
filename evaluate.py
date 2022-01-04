import sys
import logging
import argparse
import glob
import numpy as np
from pathlib import Path
from dynaconf import settings
from tqdm import tqdm

import load_model
import predict
import utils

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def evaluate(model_path, input, output):
  interpreter = load_model.load_tflite_model(model_path)
  label_id_offset = settings.LABEL_ID_OFFSET
  category_index = {'digital_class_id': {'id': 'digital_class_id', 'name': 'digital'}}

  images = glob.glob(input + '/*.jpeg', recursive=False)

  logging.info("Starting evaluation..")

  for i in tqdm(range(len(images))):
    image_path = images[i]
    gt_path = image_path.replace('.jpeg', '-bbox.txt')
    out_image_path = output + "/" + Path(image_path).stem + "-prediction.jpeg"

    image_tensor = utils.load_image_into_numpy_array(image_path)
    boxes, classes, scores = predict.invoke(interpreter, image_tensor)

    utils.plot_detections(
      image_tensor,
      boxes[0],
      classes[0].astype(np.uint32) + label_id_offset,
      scores[0],
      category_index,
      figsize=(15, 20),
      image_name=out_image_path
    )
  

def main():
  """
  Evaluate the model on testing data. 
  The function draws bounding box on testing images
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", default="meeter_detection/training_data/test", help="Folder where we have stored testing images")
  parser.add_argument("--output", default="meeter_detection/output", help="Folder where we store predicted images with bounding box")
  args = parser.parse_args()

  Path(args.output).mkdir(parents=True, exist_ok=True)
  evaluate(settings.MODEL_PATH, args.input, args.output)

if __name__=="__main__":
  main()