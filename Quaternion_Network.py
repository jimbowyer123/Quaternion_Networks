# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 09:59:28 2018

@author: jb2968
"""

import tensorflow as tf
import Quaternion_Arithmetic as QA

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
Y_train=tf.keras.utils.to_categorical(y_train,10)

lr=tf.placeholder(tf.float32,name='learning_rate')

X=tf.placeholder(tf.float32,shape=(None,28,28))
Y=tf.placeholder(tf.float32,shape=(None,10))


def Classify(X):
    XX=tf.expand_dims(X,3)
    with tf.variable_scope('class'):
        w_init=tf.contrib.layers.xavier_initializer()
        b_init=tf.zeros_initializer()
        conv1=tf.nn.conv2d(XX,tf.get_variable('conv1',shape=(5,5,1,120),initializer=w_init),[1,1,1,1],"SAME")
        conv1b=tf.add(conv1,tf.get_variable('bias1',shape=(28,28,120),initializer=b_init))
        A1=tf.nn.relu(conv1b,name='A1')
        
        QA1=tf.reshape(A1,(tf.shape(A1)[0],28*28*30,4))
        QW1=tf.get_variable('QW1',shape=(512,28*28*30,4),initializer=w_init)
        QB1=tf.get_variable('QB1',shape=(512,4))
        
        
        QZ1=QA.quaternion_fully_connected(QA1,QW1,QB1)
        
        QA2=QA.quaternion_mod_relu(QZ1,0.5)
        QW2=tf.get_variable('QW2',shape=(256,512,4),initializer=w_init)
        QB2=tf.get_variable('QB2',shape=(256,4))
        
        QZ2=QA.quaternion_fully_connected(QA2,QW2,QB2)
        
        QA3=QA.quaternion_mod_relu(QZ2,0.5)
        QW3=tf.get_variable('QW3',shape=(10,256,4),initializer=w_init)
        QB3=tf.get_variable('QB3',shape=(10,4))
        
        QZ3=QA.quaternion_fully_connected(QA3,QW3,QB3)
        
        logits=QA.quaternion_norm(QZ3)
        
        output=tf.nn.softmax(logits)
        
        return output

O=Classify(X)

Loss=tf.reduce_mean((Y-O)**2)
Optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(Loss,var_list=tf.trainable_variables())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    learning_rate=0.1
    L=[]
    for epoch in range(1000):
        print('\nEpoch: {}'.format(epoch))
        LE=0
        LB=0
        for batch in range(int(x_train.shape[0]/100)):
            _,l=sess.run([Optimizer,Loss],feed_dict={X:x_train[batch:batch+100,:,:],Y:Y_train[batch:batch+100,:],lr:learning_rate})
            LE+=l
            LB+=l
            if batch%10==0:
                print('Batch {}: {}'.format(batch,LB))
                LB=0
        L.append(LE)
        if epoch>1:
            if learning_rate<1e-8:
                pass
            elif L[-1]>=L[-2]:
                learning_rate*=0.1
                print(learning_rate)
        print('\nTotal Loss: {}'.format(LE))
        
        
        
    