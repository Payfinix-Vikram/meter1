import sys
import logging
import tensorflow as tf

IMG_SIZE = 320

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def load_tflite_model(model_path):
  """
  Load the tf-lite model into memory.
  """

  logging.info(f'Loading tflite model: {model_path}') 

  # Load the TFLite model and allocate tensors.
  interpreter = tf.lite.Interpreter(model_path=model_path)
  interpreter.allocate_tensors()

  # Note that the first frame will trigger tracing of the tf.function, which will
  # take some time, after which inference should be fast.
  # Run model through a dummy image

  preprocessed_image = tf.zeros([1, IMG_SIZE, IMG_SIZE, 3])

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  interpreter.set_tensor(input_details[0]['index'], preprocessed_image.numpy())
  interpreter.invoke()

  boxes = interpreter.get_tensor(output_details[1]['index'])
  classes = interpreter.get_tensor(output_details[3]['index'])
  scores = interpreter.get_tensor(output_details[0]['index'])

  logging.info(f'Successfully loaded tflite model: {model_path}') 

  return interpreter

def main():
  load_tflite_model('model.tflite')

if __name__=="__main__":
  main()


