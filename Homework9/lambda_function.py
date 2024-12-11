import numpy as np
from PIL import Image
from io import BytesIO
from urllib import request
import tensorflow as tf

# load model
interpreter = tf.lite.Interpreter(model_path="model_2024_hairstyle_v2.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# helper functions
def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size=(200, 200)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def lambda_handler(event, context):
    url = event['url']
    
    # prepare image
    img = download_image(url)
    img = prepare_image(img)
    
    # convert image to numpy array
    image_array = np.array(img) / 255.0 
    image_array = image_array.astype(np.float32)
    input_data = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # perform inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # return output
    return {"prediction": output_data.tolist()}
