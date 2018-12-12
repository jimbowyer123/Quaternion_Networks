# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 12:49:06 2018

@author: jb2968
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import Quaternion_Layers as QL
import Quaternion_Arithmetic as QA

tf.reset_default_graph()

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
Y_train=tf.keras.utils.to_categorical(y_train,10)

lr=tf.placeholder(tf.float32,name='learning_rate')

X=tf.placeholder(tf.float32,shape=(None,28,28))
Y=tf.placeholder(tf.float32,shape=(None,10))


XX=tf.expand_dims(X,3)
w_init=tf.contrib.layers.xavier_initializer()
b_init=tf.zeros_initializer()
with tf.variable_scope('Conv1'): 
    conv1=tf.nn.conv2d(XX,tf.get_variable('Weights',shape=(5,5,1,120),initializer=w_init),[1,1,1,1],"SAME")
    conv1b=tf.add(conv1,tf.get_variable('Bias',shape=(28,28,120),initializer=b_init))
    A1=tf.nn.relu(conv1b,name='Relu')
        
QA1=tf.reshape(A1,(tf.shape(A1)[0],28*28*120))
        
with tf.variable_scope('Dense2'):
    
    QA2=QL.Quaternion_Dense_Layer(512)(QA1)
        
    qrelu2=QL.Quaternion_Mod_Relu()(QA2)
        
with tf.variable_scope('Dense3'):
    
    QA3=QL.Quaternion_Dense_Layer(256)(qrelu2)
        
    qrelu3=QL.Quaternion_Mod_Relu()(QA3)

with tf.variable_scope('Dense4'):        
    output=QL.Quaternion_Dense_Layer(10)(qrelu3)
        
O=tf.nn.softmax(QA.quaternion_norm(tf.reshape(output,(tf.shape(output)[0],10,4))))

TV=tf.trainable_variables()
Loss=tf.reduce_mean((Y-O)**2,name='Loss')
Optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(Loss,var_list=TV,name='Train')
gradients=tf.gradients(Loss,TV)
tf.summary.scalar('loss',Loss)
merged_summary=tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    learning_rate=0.1
    writer=tf.summary.FileWriter('C:/Users/jb2968/Tensor_Board/Quaternion_Networks/Quaternion_Network')
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
    qd2=sess.run(qd2,feed_dict={X:x_train[:100,:,:]})
    print(qd2)
'''
'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    grad=sess.run(gradients,feed_dict={X:x_train[:100,:,:],Y:Y_train[:100,:]})
    print(grad)   
'''     
        
        