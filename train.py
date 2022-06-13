import tensorflow as tf
from loss import yolo_loss
from model import yolo_net
from voc_dataset import *
import datetime


def train(train_ds, val_ds, epochs=20, batch_size=32, optim="sgd", lr=0.01):
    """
    TensorFlow自定义训练
    """
    
    # 准备数据
    train_size = train_ds.cardinality().numpy()
    train_ds = train_ds.shuffle(train_size).batch(batch_size).prefetch(1)
    val_ds = val_ds.batch(batch_size).prefetch(1)

    # 构建模型
    input = tf.keras.layers.Input(shape=(224, 224, 3))
    output = yolo_net(input)
    model = tf.keras.Model(input, output)
    model.summary()
    
    # 配置优化器和自适应学习率
    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    if optim=="sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, decay=0.0005)
    if optim=="adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # 配置指标和监控
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    val_loss = tf.keras.metrics.Mean(name="val_loss")
    train_acc = tf.keras.metrics.Accuracy(name="train_acc")
    val_acc = tf.keras.metrics.Accuracy(name="val_acc")

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'runs/' + current_time + '/train'
    val_log_dir = 'runs/' + current_time + '/val'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            loss_value = yolo_loss(labels, logits)
        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss_value)

    @tf.function
    def val_step(images, labels):
        logits = model(images, training=False)
        loss_value = yolo_loss(labels, logits)
        val_loss(loss_value)


    # 训练循环
    for epoch in range(epochs):

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_acc.result(), step=epoch)
            
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', val_acc.result(), step=epoch)
        
        # 指标清零
        train_loss.reset_states()
        val_loss.reset_states()
        train_acc.reset_states()
        val_acc.reset_states()
        
        # 训练阶段
        for step, (images, labels) in enumerate(train_ds):
            train_step(images, labels)

        # 验证阶段
        for step, (images, labels) in enumerate(val_ds):
            val_step(images, labels)

        pattern = '{:.3f}'
        print(
            'Epoch ' + '{}'.format(epoch+1),
            'Loss: ' + pattern.format(train_loss.result()),
            'Accuracy: ' + pattern.format(train_acc.result()),
            'Val Loss: ' + pattern.format(val_loss.result()), 
            'Val Accuracy: ' + pattern.format(val_acc.result())
        )

    model.save("yolo.h5")



if __name__ == '__main__':


    train_txt = os.path.join(voc_txt_root, 'train.txt')
    val_txt = os.path.join(voc_txt_root, 'val.txt')

    train_image_paths, train_image_labels = generate_voc_data_list(train_txt)
    val_image_paths, val_image_labels = generate_voc_data_list(val_txt)
    
    train_ds = generate_dataset(train_image_paths, train_image_labels)
    val_ds = generate_dataset(val_image_paths, val_image_labels)
    
    print("training on {} images, validating on {} images.".format(
        train_ds.cardinality().numpy(),
        val_ds.cardinality().numpy()
    ))

    train(train_ds, val_ds, epochs=50, batch_size=64, optim="adam", lr=1e-4)