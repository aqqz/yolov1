import xml.etree.ElementTree as et
import os
import random

from utils import *




voc_annotation_root = '/home/taozhi/datasets/VOCdevkit/VOC2012/Annotations'
voc_image_root = '/home/taozhi/datasets/VOCdevkit/VOC2012/JPEGImages'
voc_txt_root = '/home/taozhi/datasets/VOCdevkit/VOC2012/ImageSets/Main'

voc_class_list = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def parse_xml(xml_path):
    """
    输入xml文件路径，返回对应图片的宽、高，以及包含的目标信息
    """
    parser = et.parse(xml_path)
    root = parser.getroot()

    size = root.find('size')
    img_w = eval(size.find('width').text)
    img_h = eval(size.find('height').text)
    
    ob_infos = []
    objs = root.findall('object')
    for obj in objs:
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = eval(bbox.find('xmin').text)
        ymin = eval(bbox.find('ymin').text)
        xmax = eval(bbox.find('xmax').text)
        ymax = eval(bbox.find('ymax').text)
        ob_id = voc_class_list.index(name)
        
        ob_infos.append([ob_id, xmin, ymin, xmax, ymax])

    return img_w, img_h, ob_infos


def encode_voc_label(img_w, img_h, ob_infos):
    """
    输入voc数据格式的目标信息，返回编码后的yolo格式的label
    """
    label = np.zeros(shape=(S, S, 5*B+C), dtype="float32")
    for ob_info in ob_infos:
        ob_id = ob_info[0]
        scale_x = img_w / img_size
        scale_y = img_h / img_size
        # scale image from origin to img_size
        xmin = ob_info[1] / scale_x
        ymin = ob_info[2] / scale_y
        xmax = ob_info[3] / scale_x
        ymax = ob_info[4] / scale_y
        # compute center coord, w, h
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        w = xmax - xmin
        h = ymax - ymin
        # compute grid offset
        grid_x = int(x_center // grid_size)
        grid_y = int(y_center // grid_size)
        # compute label value in [0, 1]
        normal_x = x_center / grid_size - grid_x
        normal_y = y_center / grid_size - grid_y
        normal_w = w / img_size
        normal_h = h / img_size
        # fill label
        ob_vector = label[grid_x, grid_y]
        ob_vector[0] = 1
        ob_vector[1:5] = [normal_x, normal_y, normal_w, normal_h]
        ob_vector[5] = 1 
        ob_vector[6:10] = [normal_x, normal_y, normal_w, normal_h]
        ob_vector[10+ob_id] = 1
    return label        



def decode_voc_label(img_w, img_h, label):
    """
    输入编码后的voc数据集label信息，返回解码后的原xml信息
    """
    ob_infos = []
    # find ob
    for i in range(S):
        for j in range(S):
            ob_info = label[i, j]
            if ob_info[0]==1:
                # get normal x, y, w, h   
                normal_x = ob_info[1]
                normal_y = ob_info[2]
                normal_w = ob_info[3]
                normal_h = ob_info[4]
                # resize x,y,w,h to img_size
                x_center = (normal_x + i) * grid_size
                y_center = (normal_y + j) * grid_size
                w = normal_w * img_size
                h = normal_h * img_size
                # get xmin, ymin, xmax, ymax in img_size
                xmin = x_center - w/2
                ymin = y_center - h/2
                xmax = x_center + w/2
                ymax = y_center + h/2
                # rescale to [img_w, img_h]
                scale_y = img_h / img_size
                scale_x = img_w / img_size
                xmin = round(xmin*scale_x)
                ymin = round(ymin*scale_y)
                xmax = round(xmax*scale_x)
                ymax = round(ymax*scale_y)
                ob_cls = ob_info[10:]
                for k in range(C):
                    if ob_cls[k] == 1:
                        ob_id = k
                        break
                ob_infos.append([ob_id, xmin, ymin, xmax, ymax])

    return ob_infos


def generate_voc_data_list(txt_file):
    """
    返回图片路径列表和构造的label列表
    """
    image_paths = []
    image_labels = []

    f = open(txt_file)
    lines = f.readlines()
    for line in lines:
        line = line[:-1]
        img_path = os.path.join(voc_image_root, line + '.jpg')
        xml_path = os.path.join(voc_annotation_root, line + '.xml')
        # parse xml
        img_w, img_h, ob_infos = parse_xml(xml_path)
        img_label = encode_voc_label(img_w, img_h, ob_infos)
        # fill list
        image_paths.append(img_path)
        image_labels.append(img_label)

    # shuffle
    if str(txt_file).endswith('train.txt'):
        random.seed(123)
        random.shuffle(image_paths)
        random.seed(123)
        random.shuffle(image_labels)


    return image_paths, image_labels


if __name__ == '__main__':

    xml_path = os.path.join(voc_annotation_root, '2007_000032.xml')
    img_w, img_h, ob_infos = parse_xml(xml_path)
    print(img_w, img_h, ob_infos)
    label = encode_voc_label(img_w, img_h, ob_infos)
    ob_infos = decode_voc_label(img_w, img_h, label)

    img_path = os.path.join(voc_image_root, '2007_000032.jpg')
    draw_box(img_path, ob_infos)