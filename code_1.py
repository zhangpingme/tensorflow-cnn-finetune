import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
rng = np.random

learning_rate = 0.001
training_epochs = 1000
display_step = 50

train_X = np.asarray()

X = tf.placeholder("float")
Y = tf.placeholder("float")

# 模型参数
W = tf.Variable(rng.rand(),name="weight")
b = tf.Variable(rng.rand(),name="bias")

# 构建线性模型
pred = tf.add(tf.matmul(X,W),b)

init = tf.initialize_all_variables

# 开始
with tf.Session as sess:
    sess,run(init)
