import tensorflow as tf
from utils import *

def compute_iou(box1, box2):
    """
    返回两个bounding box的IOU
    box1, box2: [x, y, w, h] (?, S, S, 4)
    """
    # [x, y, w, h] -> [xmin, ymin, xmax, ymax]
    box1_coord = tf.stack([
        box1[..., 0] - box1[..., 2] / 2.0,
        box1[..., 1] - box1[..., 3] / 2.0,
        box1[..., 0] + box1[..., 2] / 2.0, 
        box1[..., 1] + box1[..., 3] / 2.0, 
    ], axis=-1)
    box2_coord = tf.stack([
        box2[..., 0] - box2[..., 2] / 2.0,
        box2[..., 1] - box2[..., 3] / 2.0,
        box2[..., 0] + box2[..., 2] / 2.0, 
        box2[..., 1] + box2[..., 3] / 2.0, 
    ], axis=-1)

    # inter box (xmin, ymin, xmax, ymax)
    inter_coord = tf.stack([
        tf.maximum(box1_coord[..., 0], box2_coord[..., 0]),
        tf.maximum(box1_coord[..., 1], box2_coord[..., 1]),
        tf.minimum(box1_coord[..., 2], box2_coord[..., 2]),
        tf.minimum(box1_coord[..., 3], box2_coord[..., 3]),
    ], axis=-1)

    inter_w = tf.maximum(inter_coord[..., 2]-inter_coord[..., 0], 0)
    inter_h = tf.maximum(inter_coord[..., 3]-inter_coord[..., 1], 0)

    s_inter = inter_w*inter_h
    s_box1 = (box1_coord[..., 2]-box1_coord[..., 0])*(box1_coord[..., 3]-box1_coord[..., 1])
    s_box2 = (box2_coord[..., 2]-box2_coord[..., 0])*(box2_coord[..., 3]-box2_coord[..., 1])
    
    iou = s_inter / (s_box1 + s_box2 - s_inter + 1e-10)
    return iou

    


def yolo_loss(y_true, y_pred):
    """
    yolo损失函数
    """
    # 获取标签、预测
    batch = y_true.shape[0]
    c = tf.reshape(y_true[..., 0], shape=[batch, S, S, 1]) #(?, S, S, 1)
    
    _c = tf.reshape(tf.stack([
        y_pred[..., 0],
        y_pred[..., 5]
    ], axis=-1), shape=[batch, S, S, B]) # (?, S, S, B)

    xywh = tf.reshape(tf.stack([
        y_true[..., 1],
        y_true[..., 2],
        tf.sqrt(y_true[..., 3] + 1e-10),
        tf.sqrt(y_true[..., 4] + 1e-10),
        y_true[..., 6],
        y_true[..., 7],
        tf.sqrt(y_true[..., 8] + 1e-10),
        tf.sqrt(y_true[..., 9] + 1e-10),
    ], axis=-1), shape=(batch, S, S, B, 4)) #(?, S, S, 2, 4)
    
    _xywh = tf.reshape(tf.stack([
        y_pred[..., 1],
        y_pred[..., 2],
        tf.sqrt(y_pred[..., 3] + 1e-10),
        tf.sqrt(y_pred[..., 4] + 1e-10),
        y_pred[..., 6],
        y_pred[..., 7],
        tf.sqrt(y_pred[..., 8] + 1e-10),
        tf.sqrt(y_pred[..., 9] + 1e-10),
    ], axis=-1), shape=(batch, S, S, B, 4)) #(?, S, S, 2, 4)
    
    p = y_true[..., 10:] #(?, S, S, C)
    _p = y_pred[..., 10:] # (?, S, S, C)

    # offset
    offset_x = np.array([np.reshape(np.array([i for i in range(S)]*S), (S, S))]*batch) #(?, S, S)
    offset_y = np.array([np.reshape(np.array([i for i in range(S)]*S), (S, S)).transpose()]*batch) #(?, S, S)
    offset_x = tf.cast(offset_x, dtype=tf.float32)
    offset_y = tf.cast(offset_y, dtype=tf.float32)
    
    box = tf.reshape(tf.stack([
        (y_true[..., 1] + offset_x) / S,
        (y_true[..., 2] + offset_y) / S, 
        y_true[..., 3],
        y_true[..., 4],
        (y_true[..., 6] + offset_x) / S,
        (y_true[..., 7] + offset_y) / S, 
        y_true[..., 8],
        y_true[..., 9],
    ], axis=-1), shape=(batch, S, S, B, 4)) # (?, S, S, B, 4)

    _box = tf.reshape(tf.stack([
        (y_pred[..., 1] + offset_x) / S,
        (y_pred[..., 2] + offset_y) / S, 
        y_pred[..., 3],
        y_pred[..., 4],
        (y_pred[..., 6] + offset_x) / S,
        (y_pred[..., 7] + offset_y) / S, 
        y_pred[..., 8],
        y_pred[..., 9],
    ], axis=-1), shape=(batch, S, S, B, 4)) #( ?, S, S, B, 4)

    # compute iou
    iou_scores = compute_iou(box, _box) #(?, S, S, B)
    # tf.print(iou_scores)

    # find max iou and response box
    max_iou = tf.reduce_max(iou_scores, axis=3, keepdims=True) # (?, S, S, 1)
    # tf.print(max_iou)
    box_mask = tf.cast(iou_scores >= max_iou, dtype=tf.float32) #(?, S, S, 2)
    
    # compute object mask
    object_mask = c*box_mask #(?, S, S, 2)
    noobject_mask = 1-object_mask #(?, S, S, 2)

    # compute loss
    positive_delta = object_mask*(iou_scores - _c) #(?, S, S, 2)
    positive_loss = tf.reduce_mean(tf.reduce_sum(tf.square(positive_delta), axis=[1, 2, 3]))

    negative_delta = noobject_mask*(0-_c)
    negative_loss = tf.reduce_mean(tf.reduce_sum(tf.square(negative_delta), axis=[1, 2, 3]))

    cls_delta = c*(p-_p) #(?, S, S, 1)
    cls_loss = tf.reduce_mean(tf.reduce_sum(tf.square(cls_delta), axis=[1, 2, 3]))

    coord_mask = tf.expand_dims(object_mask, 4) #(?, S, S, B, 1)
    loc_delta = coord_mask*(xywh-_xywh) #(?, S, S, B, 4)
    loc_loss = tf.reduce_mean(tf.reduce_sum(tf.square(loc_delta), axis=[1, 2, 3, 4]))

    loss = 5*loc_loss + positive_loss + 0.5*negative_loss + cls_loss
    return loss    


if __name__ == '__main__':

    box1 = np.expand_dims(np.array([1, 1, 2, 2], dtype="float32"), axis=0)
    box2 = np.expand_dims(np.array([2, 2, 2, 2], dtype="float32"), axis=0)

    iou = compute_iou(box1, box2)
    print(iou)