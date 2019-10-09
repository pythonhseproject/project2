from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, Conv2DTranspose, MaxPooling2D, ZeroPadding2D, Dense, Flatten
from keras import backend as keras

class modelsClass(object):

    def __init__(self, img_rows = 272, img_cols = 480):

        self.img_rows = img_rows
        self.img_cols = img_cols

    
    def cnn(self):
        input_image = Input((self.img_rows, self.img_cols, 3))
        # (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

        # Normalize pixel values to be between 0 and 1
        # ValueError: Shapes (576, 64) and (1024, 64) are incompatibleinput_image = input_image / 255.0
        
        # conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_blurred)
        # model = models.Sequential()
        conv1 = Conv2D(32, (3, 3), activation='relu')(input_image)
        pool1 = MaxPooling2D((2, 2))(conv1)
        
        conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
        pool2 = MaxPooling2D((2, 2))(conv2)
        
        conv3 = Conv2D(64, (3, 3), activation='relu')(pool2)

        flatten = Flatten()(conv3)
        dense4 = Dense(64, activation='relu')(flatten)
        dense5 = Dense(10, activation='softmax')(dense4)
        
        model = Model(inputs=input_image, outputs=dense5)
        return model
    
    def classification(self):
        input_image = Input((self.img_rows, self.img_cols))
        input_shape = (self.img_rows, self.img_cols)
        flatten0 = Flatten(input_shape=input_shape)(input_image)
        dense1 = Dense(128, activation='relu')(flatten0)
        dense2 = Dense(10, activation='softmax')(dense1)

        
        
        # input_image = Input((self.img_rows, self.img_cols,3))
        # input_dim = self.img_rows*self.img_cols*3
        # 
        # dense1 = Dense(units=64, activation='relu', input_dim=input_dim)(input_image)
        # dense2 = Dense(units=10, activation='softmax')(dense1)
        
        # model = Model(inputs=input_image, outputs=dense2)
        model = Model(inputs=input_image, outputs=dense2)
        return model
    
