import gradio as gr
import tensorflow as tf
import numpy as np
import cv2

# loading the files
model_path = "fine_tuned_VGG16.h5"
model = tf.keras.models.load_model(model_path)

# Util Functions
def process_image(image):
    # Convert into tensor
    image = tf.convert_to_tensor(image)

    # Cast the image to tf.float32
    image = tf.cast(image, tf.float32)
    
    # Resize the image to img_resize
    image = tf.image.resize(image, (224,224))
    
    # Normalize the image
    image /= 255.0
    
    # Return the processed image and label
    return image

def finalize_segmentation(prediction):#;, shape):
  
  seg_img = cv2.cvtColor(prediction, cv2.COLOR_BGR2GRAY)
  th = seg_img.mean()
  _, thresholded = cv2.threshold(seg_img, th, 255, cv2.THRESH_BINARY)
  #thresholded = cv2.resize(thresholded, shape[:2])

  return np.uint8(thresholded)

def paint_transparent(mask, image):
    
    # Declerating the params
    overlay_color = (0, 255, 0)  # green
    alpha = 0.5
    
    # Adjsuting the mask shape
    shape = image.shape
    mask = cv2.resize(mask, (shape[1], shape[0]))

    # Drawing the object
    overlay = np.zeros_like(image)
    overlay[:, :, :] = overlay_color
    
    # Implementing the mask according to notation (0->object, 255->not object :> bitwise_not)
    overlay = cv2.bitwise_not(overlay, overlay, mask=mask)
    
    # transparent concatenation
    output = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    return output

def predict(image):

  # Pre-procesing the data
  images = process_image(image)
  #shape = images.shape

  # Batching
  batched_images = tf.expand_dims(images, axis=0)
  
  prediction = model.predict(batched_images)[0]

  mask = finalize_segmentation(prediction)#, shape)

  segmented_image = paint_transparent(mask, image)

  return segmented_image

# declerating the params
comp_parms = {
  "fn":predict, 
  "inputs":gr.Image(shape=(224, 224), type="numpy"),
  "outputs":gr.Image(),
  "examples":"sample_images/images"
}
demo = gr.Interface(**comp_parms)
    

# Launching the demo
if __name__ == "__main__":
    demo.launch()
