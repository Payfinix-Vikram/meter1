import sys
import logging
import tensorflow as tf

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def invoke(interpreter, image_tensor):
  """
  interpreter: A tflite interpreter
  image_tensors: A list images as tensors to predict bounding boxes

  The functions invokes the model with input images and does non max suppression on
  resulting bboxes, thus returning the best bbox as result.

  Output: A bounding box, score
  """

  if not len(image_tensor) or not interpreter:
    logging.error('Missing interpreter or image_tensor for prediction')
    return None, None

  logging.debug('Running bbox prediction on image.')

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  image_tensor_batch = tf.expand_dims(image_tensor, axis=0) # batch_size = 1
  processed_tensors = tf.image.per_image_standardization(image_tensor_batch)

  interpreter.set_tensor(input_details[0]['index'], processed_tensors)
  interpreter.invoke()

  boxes = interpreter.get_tensor(output_details[1]['index'])
  classes = interpreter.get_tensor(output_details[3]['index'])
  scores = interpreter.get_tensor(output_details[0]['index'])

  return boxes, classes, scores, 

