import os, sys
import numpy as np
import tensorflow as tf
import datetime
from model import ResNetModel
sys.path.insert(0, '../utils')
from preprocessor import BatchPreprocessor

learning_rate = 0.0001  # Learning rate for adam optimizer  学习率
resnet_depth = 50       # ResNet架构的层数
num_epochs = 10         # 训练的批次数
batch_size = 32         # 每批次的样本数目
log_step = 10           # Logging period in terms of iteration

training_file = '../data/train.txt'  # Training dataset file  训练集的位置
val_file = '../data/val.txt'         # Validation dataset file  验证集的位置
tensorboard_root_dir = '../training' # Root directory to put the training logs and weights  模型和权重的存储位置

# height为传入图片的高度，width为传入图片的宽度，num_classes为分类数
def main(height,width,num_classes):
    train_layers = 'fc'  # finetuning layers, seperated by commas
    multi_scale = ''     # As preprocessing; scale the image randomly between 2 numbers and crop randomly at network\'s input size

    # Create training directories
    now = datetime.datetime.now()
    # 格式化输出
    train_dir_name = now.strftime('resnet_%Y%m%d_%H%M%S')
    train_dir = os.path.join(tensorboard_root_dir, train_dir_name)
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')
    tensorboard_dir = os.path.join(train_dir, 'tensorboard')
    tensorboard_train_dir = os.path.join(tensorboard_dir, 'train')
    tensorboard_val_dir = os.path.join(tensorboard_dir, 'val')

    # 如果不存在这个目录，则新建这样的一个目录  if not os.path.isdir()    os.mkdir()
    if not os.path.isdir(tensorboard_root_dir): os.mkdir(tensorboard_root_dir)
    if not os.path.isdir(train_dir): os.mkdir(train_dir)
    if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
    if not os.path.isdir(tensorboard_dir): os.mkdir(tensorboard_dir)
    if not os.path.isdir(tensorboard_train_dir): os.mkdir(tensorboard_train_dir)
    if not os.path.isdir(tensorboard_val_dir): os.mkdir(tensorboard_val_dir)

    # Write flags to txt
    flags_file_path = os.path.join(train_dir, 'flags.txt')
    flags_file = open(flags_file_path, 'w')
    flags_file.write('learning_rate={}\n'.format(learning_rate))
    flags_file.write('resnet_depth={}\n'.format(resnet_depth))
    flags_file.write('num_epochs={}\n'.format(num_epochs))
    flags_file.write('batch_size={}\n'.format(batch_size))
    flags_file.write('train_layer={}\n'.format(train_layers))
    flags_file.write('multi_scale={}\n'.format(multi_scale))
    flags_file.write('tensorboard_root_dir={}\n'.format(tensorboard_root_dir))
    flags_file.write('log_step={}\n'.format(log_step))
    flags_file.close()

    # Placeholders，占位符，x是输入数据的占位符，shape=(batch_size,height,width,3)
    x = tf.placeholder(tf.float32, [batch_size, height, width, 3])
    # 占位符，y是输出标签的占位符，shape=[样本数，类别数]
    y = tf.placeholder(tf.int8, [batch_size, num_classes])
    # 是否正在训练的变量，其变量类型是bool，为空
    is_training = tf.placeholder('bool', [])

    # Model
    train_layers = train_layers.split(',')  # 没看懂
    model = ResNetModel(is_training, depth=resnet_depth, num_classes=num_classes)
    loss = model.loss(x, y)
    train_op = model.optimize(learning_rate, train_layers)

    # Training accuracy of the model
    correct_pred = tf.equal(tf.argmax(model.prob, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # 概要
    tf.summary.scalar('train_loss', loss)
    tf.summary.scalar('train_accuracy', accuracy)
    merged_summary = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(tensorboard_train_dir)
    val_writer = tf.summary.FileWriter(tensorboard_val_dir)
    saver = tf.train.Saver()

    # Batch preprocessors
    multi_scale = multi_scale.split(',')
    if len(multi_scale) == 2:
        multi_scale = [int(multi_scale[0]), int(multi_scale[1])]
    else:
        multi_scale = None

    train_preprocessor = BatchPreprocessor(dataset_file_path=training_file, num_classes=num_classes,
                                           output_size=[100, 100], horizontal_flip=True, shuffle=True, multi_scale=multi_scale)
    val_preprocessor = BatchPreprocessor(dataset_file_path=val_file, num_classes=num_classes, output_size=[100, 100])

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = np.floor(len(train_preprocessor.labels) / batch_size).astype(np.int16)
    val_batches_per_epoch = np.floor(len(val_preprocessor.labels) / batch_size).astype(np.int16)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer.add_graph(sess.graph)

        # Load the pretrained weights
        model.load_original_weights(sess, skip_layers=train_layers)

        # Directly restore (your model should be exactly the same with checkpoint)
        # saver.restore(sess, "/Users/dgurkaynak/Projects/marvel-training/alexnet64-fc6/model_epoch10.ckpt")

        print("{} Start training...".format(datetime.datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.datetime.now(), tensorboard_dir))

        for epoch in range(num_epochs):
            print("{} Epoch number: {}".format(datetime.datetime.now(), epoch+1))
            step = 1

            # Start training
            while step < train_batches_per_epoch:
                batch_xs, batch_ys = train_preprocessor.next_batch(batch_size)
                sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys, is_training: True})

                # Logging
                if step % log_step == 0:
                    s = sess.run(merged_summary, feed_dict={x: batch_xs, y: batch_ys, is_training: False})
                    train_writer.add_summary(s, epoch * train_batches_per_epoch + step)

                step += 1

            # Epoch completed, start validation
            print("{} Start validation".format(datetime.datetime.now()))
            test_acc = 0.
            test_count = 0

            for _ in range(val_batches_per_epoch):
                batch_tx, batch_ty = val_preprocessor.next_batch(batch_size)
                acc = sess.run(accuracy, feed_dict={x: batch_tx, y: batch_ty, is_training: False})
                test_acc += acc
                test_count += 1

            test_acc /= test_count
            s = tf.Summary(value=[
                tf.Summary.Value(tag="validation_accuracy", simple_value=test_acc)
            ])
            val_writer.add_summary(s, epoch+1)
            print("{} Validation Accuracy = {:.4f}".format(datetime.datetime.now(), test_acc))

            # Reset the dataset pointers
            val_preprocessor.reset_pointer()
            train_preprocessor.reset_pointer()

            print("{} Saving checkpoint of model...".format(datetime.datetime.now()))

            #save checkpoint of the model
            checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch'+str(epoch+1)+'.ckpt')
            save_path = saver.save(sess, checkpoint_path)

            print("{} Model checkpoint saved at {}".format(datetime.datetime.now(), checkpoint_path))

if __name__ == '__main__':
    main(100,100,2)
