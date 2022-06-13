import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Dense, LeakyReLU, Flatten, Reshape, Dropout
from utils import *
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet import MobileNet

# 不能在imagenet上训练darknet，弃用
def conv_leaky(input, filters, kernel_size, strides=1, padding="same"):
    x = Conv2D(filters, kernel_size, strides, padding)(input)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def darknet_backbone(input):
    x = conv_leaky(input, 64, 7, 2)
    x = MaxPooling2D(2, 2)(x)

    x = conv_leaky(x, 192, 3)
    x = MaxPooling2D(2, 2)(x)

    x = conv_leaky(x, 128, 1)
    x = conv_leaky(x, 256, 3)
    x = conv_leaky(x, 256, 1)
    x = conv_leaky(x, 512, 3)
    x = MaxPooling2D(2, 2)(x)

    for i in range(4):
        x = conv_leaky(x, 256, 1)
        x = conv_leaky(x, 512, 3)
    x = conv_leaky(x, 512, 1)
    x = conv_leaky(x, 1024, 3)
    x = MaxPooling2D(2, 2)(x)

    for i in range(2):
        x = conv_leaky(x, 512, 1)
        x = conv_leaky(x, 1024, 3)
    x = conv_leaky(x, 1024, 3)
    x = conv_leaky(x, 1024, 3, 2)

    x = conv_leaky(x, 1024, 3)
    x = conv_leaky(x, 1024, 3)

    x = conv_leaky(x, 256, 3)

    return x



# base_model = VGG16(
#     include_top=False,
#     weights='imagenet',
#     input_shape=(224, 224, 3),
# )

base_model = MobileNet(
    input_shape=(224, 224, 3),
    alpha=0.25,
    include_top=False,
    weights='imagenet',
)

base_model.trainable = True

def yolo_net(input):

    x = base_model(input)
    # x = Conv2D(256, 3, 1, padding="same", activation="relu")(x)
    x = Flatten()(x)
    x = Dense(256)(x) # 降维，防止参数过多
    x = Dropout(0.5)(x)
    x = Dense(S*S*(5*B+C), activation="sigmoid")(x)
    x = Reshape(target_shape=(S, S, 5*B+C))(x)

    return x


if __name__ == '__main__':

    input = tf.keras.layers.Input(shape=(224, 224, 3))
    output = yolo_net(input)
    model = tf.keras.Model(input, output)
    model.summary()