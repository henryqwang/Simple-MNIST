import os
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#Initizalization
X = tf.placeholder(tf.float32, [None, 784]) #First dim is left 'None', as it represents number of images in a training batch, which will be known at run time
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

init = tf.global_variables_initializer()

#Model: single layer of neurons using softmax as activation function
Y = tf.nn.softmax(tf.matmul(X, W) + b)

#Place holder for correct answer, one-hot encoded
Y_actual = tf.placeholder(tf.float32, [None, 10])

#Cost function
cross_entropy = - tf.reduce_sum(Y_actual * tf.log(Y))

#Computation of percentage of correct answers
correct_percentage = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_percentage, tf.float32))

#Set up optimizer with Gradient Descent
learning_rate = 0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy)

sess = tf.Session()
sess.run(init)

#Perform gradient descent
for i in range(2000):

    #Loading training data
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data = {X: batch_X, Y_actual: batch_Y}

    #Train
    sess.run(train_step, feed_dict=train_data)

    #Success rate for training data
    a,c = sess.run([accuracy, cross_entropy], feed_dict=train_data)

    #Success rate for test data
    test_data={X:mnist.test.images, Y_actual:mnist.test.labels}
    a,c = sess.run([accuracy, cross_entropy], feed_dict=test_data)


#Prediction accuracy after training
print("Accuracy: ")
print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y_actual: mnist.test.labels}))
