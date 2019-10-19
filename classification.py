import os, os.path, errno
import argparse
from PIL import Image
import numpy as np
from keras.models import Model
from keras.preprocessing.image import array_to_img
from models import modelsClass

def main():
    # Parse input arguments
    desc = "Classification by dense layers"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i', '--input', type=str, metavar='', required=True, help='input folder')
    args = parser.parse_args()
    inpath = args.input


    image_path = inpath + '/'
    image_names = os.listdir(image_path)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    for fname in image_names:

        img = Image.open(image_path + fname)
        img_np = (1./255)*np.array(img)[:,:,0]

        width, height = img.size
        models = modelsClass(height,width)
        model = models.classification()
        model.load_weights("checkpoints/classification.h5")

        x = np.reshape(img_np,[1,height,width])
        prediction = model.predict(x, batch_size=1,verbose=0,steps=None)
        print("On picture '%s': " %(fname) + class_names[np.argmax(prediction)])

    print("done")

if __name__ == "__main__":
    main()







