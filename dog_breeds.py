import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import itertools
import shutil
import random
import glob
import os


def load_and_prep_image(filename, img_shape=224, scale=False):
  """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (224, 224, 3).
  """
  # Read in the image
  img = tf.io.read_file(filename)
  # Decode it into a tensor
  img = tf.image.decode_jpeg(img)
  # Resize the image
  img = tf.image.resize(img, [img_shape, img_shape])
  # Rescale the image (get all values between 0 and 1)
  if scale:
    return img/255.
  else:
    return img


st.title('Dog Breeds Classification')
st.markdown("""This app is to help people who love some breeds of dogs but don't know their breed name""")
st.markdown('You can get the name of dog breed using an image. Just upload an image')


from PIL import Image

def load_image(image_file):
	img = Image.open(image_file)
	return img

...


st.subheader("Image")
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

model = keras.models.load_model('C:/Users/User PC/120_dog_breeds.hdf5')

train_dir = 'D:/ratan/dog breeds/train'
test_dir = 'D:/ratan/dog breeds/test'
valid_dir = 'D:/ratan/dog breeds/valid'

IMG_SIZE = (224, 224)

def create_data_loaders(train_dir, valid_dir, test_dir, image_size=IMG_SIZE):
  """
  Creates a training and test image BatchDataset from train_dir and test_dir.
  """
  train_data = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                  label_mode="categorical",
                                                                  image_size=image_size)
  valid_data = tf.keras.preprocessing.image_dataset_from_directory(valid_dir,
                                                                  label_mode="categorical",
                                                                  image_size=image_size)
  # Note: the test data is the same as the previous experiment, we could
  # skip creating this, but we'll leave this here to practice.
  test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                  label_mode="categorical",
                                                                  image_size=image_size)
  
  return train_data, test_data


train_data, valid_data = create_data_loaders(train_dir=train_dir, valid_dir = valid_dir, test_dir = test_dir)
class_names = train_data.class_names

if image_file is not None:

  # To See details
    file_details = {"filename":image_file.name, "filetype":image_file.type,
                              "filesize":image_file.size}
    #st.write(file_details)

              # To View Uploaded Image
    st.image(load_image(image_file))
    

    
    with open(os.path.join("D:/train",image_file.name),"wb") as f:
        f.write((image_file).getbuffer())
    img = load_and_prep_image("D:/train" + '/' + image_file.name)
    expanded = tf.expand_dims(img, axis=0) # expand image dimensions (224, 224, 3) -> (1, 224, 224, 3)
    pred = model.predict(expanded)
    st.write('This is' + ' ' + class_names[tf.argmax(pred[0])])
        #st.success("File Saved")

    



option = st.selectbox(
     'Is this prediction correct?',
     ('', 'Yes', 'No'))

if option != '':
    st.write('Thank you for your feedback')