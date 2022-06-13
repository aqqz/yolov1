import tensorflow as tf
import cv2
from voc_dataset import *



def post_progress(img_w, img_h, output_tensor):
    """
    yolo目标检测后处理
    """
    output_tensor = output_tensor[0]
    ob_infos = []
    for i in range(S):
        for j in range(S):
            grid_vector = output_tensor[i, j]
            # tf.print(grid_vector)
            box_conf = tf.stack([grid_vector[0], grid_vector[5]], axis=-1) # (2, )
            max_conf = tf.reduce_max(box_conf, keepdims=True) #(1, )
            tf.print(max_conf)
            box_mask = tf.cast(box_conf >= max_conf, dtype=tf.float32) #(2, )
            
            box_x = tf.reduce_sum(box_mask*tf.stack([grid_vector[1], grid_vector[6]], axis=-1)).numpy()
            box_y = tf.reduce_sum(box_mask*tf.stack([grid_vector[2], grid_vector[7]], axis=-1)).numpy()
            box_w = tf.reduce_sum(box_mask*tf.stack([grid_vector[3], grid_vector[8]], axis=-1)).numpy()
            box_h = tf.reduce_sum(box_mask*tf.stack([grid_vector[4], grid_vector[9]], axis=-1)).numpy()
            # tf.print(box_x, box_y, box_w, box_h)
            box_cls = grid_vector[10:]
            box_id = tf.argmax(box_cls).numpy()
            # tf.print(box_id)
            x_center = (box_x + i) * grid_size
            y_center = (box_y + j) * grid_size
            w = box_w * img_size
            h = box_h * img_size
            scale_x = img_w / img_size
            scale_y = img_h / img_size
            xmin = x_center - w/2
            ymin = y_center - h/2
            xmax = x_center + w/2
            ymax = y_center + h/2

            xmin = round(xmin*scale_x)
            ymin = round(ymin*scale_y)
            xmax = round(xmax*scale_x)
            ymax = round(ymax*scale_y)

            if max_conf >= 0.4:
                ob_infos.append([box_id, xmin, ymin, xmax, ymax])
    
    return ob_infos
            



def predict(img_path):
    """
    测试阶段
    """
    img = load_image(img_path)
    input = tf.expand_dims(img, axis=0)
    model = tf.keras.models.load_model("yolo_vgg16.h5")
    output = model.predict(input)
    
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[0:2]
    ob_infos = post_progress(img_w, img_h, output)
    draw_box(img_path, ob_infos)



if __name__ == '__main__':
    test_img = os.path.join(voc_image_root, '2008_000054.jpg')
    predict(test_img)
    