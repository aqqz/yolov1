import tensorflow as tf
import numpy as np
import time
import os
from keras.preprocessing.image import image
from voc_dataset import *

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def representative_data_gen():
    """
    选择表征数据集
    """
    for input_value in tf.data.Dataset.from_tensor_slices(test_images).batch(1).take(100):
        # Model has only one input so each data point has one element.
        yield [input_value]

def lite_convert(model_path, quantization="none", save_path="model/model.tflite"):
    """
    h5 -> tflite
    """
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantization=='int8':
        print("quantizing model by int8...\n")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.target_spec.supported_types = [tf.int8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    print("converting model...\n")
    tflite_model = converter.convert()  

    with open(save_path, "wb") as f:
        f.write(tflite_model)


def evaluate_tflite(model_path, test_images, test_labels):
    """
    在测试数据集上评估tflite模型的精度
    """
    print("evaluating tflite model...\n")
    start = time.time()
    # 创建解释器
    interpreter = tf.lite.Interpreter(model_path)
    # 分配张量
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # tf.print(input_details)
    # tf.print(output_details)

    count = 0
    accuracy = tf.keras.metrics.Accuracy()

    if input_details["dtype"] == np.uint8:
        input_scale, input_zero_point = input_details["quantization"]
        test_images = test_images / input_scale + input_zero_point

    for test_image in test_images:
        input_data = np.expand_dims(test_image, axis=0).astype(input_details['dtype'])
        # print(input_data)
        test_label = test_labels[count]
        # print(test_label)
        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details['index'])[0]
        print(output_data)
        # 计算分类精度
        accuracy.update_state(tf.argmax(test_label[..., 10:], axis=2), tf.argmax(output_data[..., 10:], axis=2))
        count += 1

    end = time.time()
    print("test tflite on: {} examples, accuracy: {}, time: {}".format(
        count, accuracy.result().numpy(), end-start
    ))

    # 清零
    count = 0
    accuracy.reset_states()


def save_samples(x_test, y_test, len, mode='gray'):
    """
    保存指定数量的测试样本
    """
    print(f"generate {len} samples for quantize.")
    x_quant = x_test[:len]
    y_quant = y_test[:len]
    count=0
    if os.path.exists('samples') == False:
        os.mkdir('samples')
    for i in x_quant:
        label = np.argmax(y_quant[count])
        image.save_img('samples/' + str(count) + '_' + str(label) + '.pgm', i, mode=mode)
        count += 1

    print("saved samples in samples/")


if __name__ == '__main__':

    val_txt = os.path.join(voc_txt_root, 'val.txt')
    val_image_paths, val_image_labels = generate_voc_data_list(val_txt)
    val_ds = generate_dataset(val_image_paths, val_image_labels)
    test_images, test_labels = load_data(val_ds) #导入验证集数据

    lite_convert('yolo.h5', quantization="int8", save_path="yolo.tflite")

    save_samples(test_images, test_labels, len=100, mode="gray")
    
    evaluate_tflite(model_path="yolo.tflite", test_images=test_images, test_labels=test_labels)