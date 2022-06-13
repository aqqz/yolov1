import tensorflow as tf
import cv2
import numpy as np


S = 7
B = 2
C = 20
img_size = 224
grid_size = img_size // S


color_list = np.random.randint(0, 255, size=[C, 3])

def load_image(img_path):
    """
    读入图像路径，返回img_tensor
    """
    raw = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, size=[224, 224])
    img = tf.cast(img, dtype=tf.float32)
    img /= 255.0
    return img


def draw_box(img_path, ob_infos):
    """
    根据输入图像和bounding box信息画框
    """
    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for ob_info in ob_infos:
        ob_id = ob_info[0]
        xmin = ob_info[1]
        ymin = ob_info[2]
        xmax = ob_info[3]
        ymax = ob_info[4]
        # choose a color
        color = (int(color_list[ob_id][0]), int(color_list[ob_id][1]), int(color_list[ob_id][2]))
        # draw bounding box
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness=2, lineType=1)
        img = cv2.putText(img, str(ob_id), (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color=color, thickness=1)
    cv2.imwrite('test.jpg', img)



def generate_dataset(image_paths, image_labels):
    """
    根据图片列表和标签列表产生tensroflow数据集
    """
    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths).map(load_image)
    label_dataset = tf.data.Dataset.from_tensor_slices(image_labels)
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))

    return dataset



if __name__ == '__main__':

    img_path = '/home/taozhi/datasets/VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg'
    img_tensor = load_image(img_path)
    print(img_tensor)
    print(tuple(color_list[0]))