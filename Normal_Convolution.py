# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 13:26:34 2018

@author: jb2968
"""

import tensorflow as tf
import Quaternion_Arithmetic as QA
import numpy as np

tf.reset_default_graph()

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
Y_train=tf.keras.utils.to_categorical(y_train,10)


lr=tf.placeholder(tf.float32)

X=tf.placeholder(tf.float32, shape=(None,28,28),name='X')
Y=tf.placeholder(tf.float32,shape=(None,10),name='Y')


XX=tf.expand_dims(X,3)
w_init=tf.contrib.layers.xavier_initializer()
b_init=tf.zeros_initializer()
with tf.variable_scope('Conv1'):
    conv1=tf.nn.conv2d(XX,tf.get_variable('Weights',shape=(5,5,1,120),initializer=w_init),[1,1,1,1],"SAME")
    conv1b=tf.add(conv1,tf.get_variable('bias',shape=(28,28,120),initializer=b_init))
    A1=tf.nn.relu(conv1b,name='Relu')
                         
FA1=tf.layers.flatten(A1)
with tf.variable_scope('Dense2'):
    FC1=tf.layers.dense(FA1,512,activation=None,kernel_initializer=w_init,name='FC2')
    A2=tf.nn.relu(FC1,name='Relu')

with tf.variable_scope('Dense3'):
      
     FC2=tf.layers.dense(A2,256,activation=None,kernel_initializer=w_init,name='FC3')
     A3=tf.nn.relu(FC2,name='A3')

with tf.variable_scope('Dense4'):  
    FC3=tf.layers.dense(A3,10,activation=None,kernel_initializer=w_init,name='FC4')

output=tf.nn.softmax(FC3,name='Output')
        
        
O=output
TV=tf.trainable_variables()
Loss=tf.reduce_mean((Y-O)**2,name='Loss')
tf.summary.scalar('loss',Loss)
Optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(Loss,var_list=TV)
gradients=tf.gradients(Loss,TV)
merged_summary=tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    learning_rate=0.1
    writer=tf.summary.FileWriter('C:/Users/jb2968/Tensor_Board/Quaternion_Networks/Normal_Convolution')
    writer.add_graph(sess.graph)
    for batch in range(int(x_train.shape[0]/100)):
        print(batch)
        _=sess.run(Optimizer,feed_dict={X:x_train[batch:batch+100,:,:],Y:Y_train[batch:batch+100,:],lr:learning_rate})
        if batch%50==0:
            ms=sess.run(merged_summary,feed_dict={X:x_train[batch:batch+100,:,:],Y:Y_train[batch:batch+100,:]})
            writer.add_summary(ms,batch)
        
    

'''
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
'''
'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    grad,c=sess.run([gradients,conv1b],feed_dict={X:x_train[:100,:,:],Y:Y_train[:100,:]})
    print(c[0,:,:,50])   
        
'''