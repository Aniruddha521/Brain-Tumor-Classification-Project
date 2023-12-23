from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D,Conv2DTranspose, Add, ReLU, Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
import tensorflow as tf
from tqdm import tqdm



def preprocess_image(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.per_image_standardization(image)
    
    return image, label

class inception_module():
    def __init__(self, filters, **kwargs):
        super(inception_module, self).__init__(**kwargs)
        self.filters = filters
        # First middle layer
        self.conv_1 = Conv2D(filters[0], (1, 1), padding='same', activation='relu')
        ###########################################################################
        # Second middle layer
        self.conv_2 = Conv2D(filters[1], (1, 1), padding='same', activation='relu')
        self.conv_3 = Conv2D(filters[2], (3, 3), padding='same', activation='relu')
        ############################################################################
        # Third middle layer
        self.conv_4 = Conv2D(filters[3], (1, 1), padding='same', activation='relu')
        self.conv_5 = Conv2D(filters[4], (5, 5), padding='same', activation='relu')
        ##############################################################################
        # Fourth middle layer
        self.maxpool_build = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
        self.conv_6 = Conv2D(filters[5], (1, 1), padding='same', activation='relu')
    def get_config(self):
        config = super(inception_module, self).get_config()
        config.update({'filters': self.filters})
        return config
    def call(self,x):
        conv1 = self.conv_1(x)
        conv2 = self.conv_2(x)
        conv3 = self.conv_3(conv2)
        conv4 = self.conv_4(x)
        conv5 = self.conv_5(conv4)
        maxpool = self.maxpool_build(x)
        conv6 = self.conv_6(maxpool)

        inception_block = concatenate([conv1, conv3, conv5, conv6], axis=-1)
        return inception_block

class residual_block(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(residual_block, self).__init__(**kwargs)
        self.filters = filters
        element_sum = filters[0] + filters[2] + filters[4] + filters[5]
        self.conv_shortcut = Conv2D(element_sum, (1, 1), padding='same')
        self.norm = tf.keras.layers.BatchNormalization()
        self.mid_conv_1 = Conv2D(element_sum, (3, 3), padding='same', activation='relu')
        self.mid_conv_2 = Conv2D(element_sum, (5, 5), padding='same', activation='relu')
        self.inception_1 = inception_module(filters=filters)
        self.inception_2 = inception_module(filters=filters)
        self.inception_3 = inception_module(filters=filters)
    def get_config(self):
        config = super(residual_block, self).get_config()
        config.update({'filters': self.filters})
        return config
    @tf.function(experimental_relax_shapes=True)
    def call(self, x):
        shortcut = self.conv_shortcut(x)

        # Layer 1
        incep_1 = self.inception_1.call(x)
        inception_1_norm = self.norm(incep_1)
        x = ReLU()(inception_1_norm)

        # Layer 2
        mid_layer_1 = self.mid_conv_1(x)
        mid_layer_2 = self.mid_conv_2(mid_layer_1)
        mid_norm = self.norm(mid_layer_2)
        x = ReLU()(mid_norm)

        # Layer 3
        incep_3 = self.inception_3.call(x)
        inception_3_norm = self.norm(incep_3)
        x = ReLU()(inception_3_norm)

        # Output
        x = Add()([x, shortcut])
        x = ReLU()(x)
        return x


model_name = input("Name of your model: ")
training_path = input("Enter path of your data set: ")
epochs = int(input("Enter the number of Epochs: "))
batch = int(input("Enter batch size: "))
execute = True

will = input("Do you want to train a model it may take some time(y/n)")
if will == "n" or will == "N":
    execute = False
if execute:
    training_data, validation_data =  tqdm(tf.keras.utils.image_dataset_from_directory(
                                                                                        training_path,
                                                                                        validation_split = 0.2,
                                                                                        subset = "both",
                                                                                        labels ='inferred',
                                                                                        label_mode = "int",
                                                                                        color_mode = "grayscale",
                                                                                        shuffle = True,
                                                                                        seed = 123,
                                                                                        image_size = (250,250),
                                                                                        batch_size = batch
                                                                                        ),
                                                                            desc="Loading Dataset",
                                                                            unit="batch")

    print("Completed loading traning data and validation data")

    training_data = training_data.map(preprocess_image)
    validation_data = validation_data.map(preprocess_image)

    # Model architecture
    inputs = tf.keras.Input((256,256,1))
    # Down scaling
    conv1 = Conv2D(16,(3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(inputs)
    conv1 = residual_block(filters=[4,16,4,16,4,4])(conv1)
    conv1 = Dropout(0.1)(conv1)
    # conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = Conv2D(16,(3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(conv1)
    pool1 = MaxPooling2D((2,2), padding = "same")(conv1)
    conv2 = Conv2D(32,(3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(pool1)
    conv2 = residual_block(filters=[8,32,8,32,8,8])(conv2)
    conv2 = Dropout(0.2)(conv2)
    # conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = Conv2D(32,(3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(conv2)
    pool2 = MaxPooling2D((2,2), padding = "same")(conv2)
    conv3 = Conv2D(64,(3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(pool2)
    conv3 = residual_block(filters=[16,64,16,64,16,16])(conv3)
    conv3 = Dropout(0.2)(conv3)
    # conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = Conv2D(64,(3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(conv3)
    pool3 = MaxPooling2D((2,2), padding = "same")(conv3)
    conv4 = Conv2D(128,(3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(pool3)
    conv4 = residual_block(filters=[32,128,32,128,32,32])(conv4)
    # conv4 = Dropout(0.1)(conv4)
    # conv4 = tf.keras.layers.BatchNormalization()(conv4)
    conv4 = Conv2D(128,(3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(conv4)
    pool4 = MaxPooling2D((2,2), padding = "same")(conv4)
    conv5 = Conv2D(256,(3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(pool4)
    conv5 = residual_block(filters=[64,256,64,256,64,64])(conv5)
    # conv5 = Dropout(0.1)(conv5)
    # conv5 = tf.keras.layers.BatchNormalization()(conv5)
    conv5 = Conv2D(256,(3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(conv5)
    # Upscaling
    up6 = Conv2DTranspose(128,(2,2), strides = (2,2), padding = "same")(conv5)
    up6 = tf.keras.layers.concatenate([up6,conv4])
    conv6 = Conv2D(128,(3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(up6)
    conv6 = residual_block(filters=[32,128,32,128,32,32])(conv6)
    # conv6 = Dropout(0.1)(conv6)
    # conv6 = tf.keras.layers.BatchNormalization()(conv6)
    conv6 = Conv2D(128,(3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(conv6)
    up7 = Conv2DTranspose(64,(2,2), strides = (2,2), padding = "same")(conv6)
    up7 = tf.keras.layers.concatenate([up7,conv3])
    conv7 = Conv2D(64,(3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(up7)
    conv7 = residual_block(filters=[16,64,16,64,16,16])(conv7)
    # conv7 = Dropout(0.1)(conv7)
    # conv7 = tf.keras.layers.BatchNormalization()(conv7)
    conv7 = Conv2D(64,(3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(conv7)
    up8 = Conv2DTranspose(32,(2,2), strides = (2,2), padding = "same")(conv7)
    up8 = tf.keras.layers.concatenate([up8,conv2])
    conv8 = Conv2D(32,(3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(up8)
    conv8 = residual_block(filters=[8,32,8,32,8,8])(conv8)
    # conv8 = Dropout(0.1)(conv8)
    # conv8 = tf.keras.layers.BatchNormalization()(conv8)
    conv8 = Conv2D(32,(3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(conv8)
    up9 = Conv2DTranspose(16,(2,2), strides = (2,2), padding = "same")(conv8)
    up9 = tf.keras.layers.concatenate([up9,conv1])
    conv9 = Conv2D(16,(3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(up9)
    conv9 = residual_block(filters=[4,16,4,16,4,4])(conv9)
    # conv9 = Dropout(0.1)(conv9)
    # conv9 = tf.keras.layers.BatchNormalization()(conv9)
    conv9 = Conv2D(16,(3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(conv9)
    # conv9 = Conv2D(1,(1,1), activation = "relu")(conv9)
    drop =Dropout(0.1)(conv9)
    flap = Flatten()(drop)
    den = Dense(40,activation = "relu")(flap)
    output = Dense(4,activation = "softmax")(den)
    model = tf.keras.models.Model(inputs,output)
    model.compile(optimizer="adam", loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    model.summary()


    checkpointer = tf.keras.callbacks.ModelCheckpoint("Tumor_detector.h5", verbose=1, save_best_only=True)
    callbacks = [
                    tf.keras.callbacks.EarlyStopping(patience=3,monitor="val_loss"),
                    tf.keras.callbacks.TensorBoard(log_dir="logs")
                ]
    history = model.fit(training_data,
                        validation_data=validation_data,
                        epochs=epochs,
                        callbacks=callbacks,
                        batch_size=16)
    save_will = input("Do you want to save model(y/n): ")
    if save_will == "n" or save_will == "N":
        model.save(model_name,overwrite=True)
else:
    exit()
