import tensorflow as tf
import cv2
from voc_dataset import *

iou_threshold = 0.5
con_threshold = 0.4

def bbox_trans(box_x, box_y, box_w, box_h, offset_x, offset_y, img_w, img_h):
    """
    处理bounding box
    [x, y, w, h] -> [xmin, ymin, xmax, ymax]
    """
    # resize output to [img_size, img_size]
    x_center = (box_x + offset_x) * grid_size
    y_center = (box_y + offset_y) * grid_size
    w = box_w * img_size
    h = box_h * img_size
    # resize output to [img_w, img_h]
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

    return xmin, ymin, xmax, ymax


def iou(box1, box2):
    """
    计算两个bounding box的iou值
    box1: [xmin, ymin, xmax, ymax]
    box2: [xmin, ymin, xmax, ymax]
    """
    s1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    s2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    
    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])

    w = max(xmax-xmin, 0)
    h = max(ymax-ymin, 0)
    s3 = w*h

    return s3 / (s1 + s2 - s3 + 1e-10)



def nms(bbox_before):
    """
    非极大值抑制，滤除多余的bounding box
    """
    bbox_after = []
    # 候选框按置信度降序排序
    bbox_before.sort(key=lambda x:x[0], reverse=True)

    # 当候选框不空
    while(len(bbox_before)!=0):
        # 输出框加入置信度最高的框
        bbox_after.insert(0, bbox_before[0])
        # 候选框删除最高的框
        bbox_before.remove(bbox_before[0])
        # 遍历候选框集合，删除与输出框iou大于阈值的候选框
        for bbox in bbox_before:
            test_iou = iou(bbox_after[0][3:], bbox[3:])
            if test_iou >= iou_threshold:
                bbox_before.remove(bbox)

    return bbox_after

    
    


def post_progress(img_w, img_h, output_tensor):
    """
    yolo目标检测后处理
    """
    output_tensor = output_tensor[0]
    bbox_before = []
    bbox_after = []
    ob_infos = []
    for i in range(S):
        for j in range(S):
            grid_vector = output_tensor[i, j]
            box1_conf = grid_vector[0]
            box1_x = grid_vector[1]
            box1_y = grid_vector[2]
            box1_w = grid_vector[3]
            box1_h = grid_vector[4]
            box2_conf = grid_vector[5]
            box2_x = grid_vector[6]
            box2_y = grid_vector[7]
            box2_w = grid_vector[8]
            box2_h = grid_vector[9]
            box_cls = grid_vector[10:]
            box_id = tf.argmax(box_cls).numpy()
            box_prob = box_cls[box_id]
            if box1_conf >= con_threshold:
                box1_xmin, box1_ymin, box1_xmax, box1_ymax = bbox_trans(box1_x, box1_y, box1_w, box1_h, i, j, img_w, img_h)
                bbox_before.append([box1_conf, box_id, box_prob, box1_xmin, box1_ymin, box1_xmax, box1_ymax])
            if box2_conf >= con_threshold:
                box2_xmin, box2_ymin, box2_xmax, box2_ymax = bbox_trans(box2_x, box2_y, box2_w, box2_h, i, j, img_w, img_h)
                bbox_before.append([box2_conf, box_id, box_prob, box2_xmin, box2_ymin, box2_xmax, box2_ymax])
            
    
    bbox_after = nms(bbox_before)
    tf.print(bbox_after)

    for bbox in bbox_after:
        ob_infos.append([bbox[1], bbox[3], bbox[4], bbox[5], bbox[6]])
    
    tf.print(ob_infos)

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
    # test_img = os.path.join(voc_image_root, '2008_000054.jpg')
    test_img = 'netpic.jpg'
    predict(test_img)
    