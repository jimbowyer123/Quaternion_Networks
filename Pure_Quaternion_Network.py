# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 10:26:05 2018

@author: jb2968
"""

import tensorflow as tf
import Pure_Quaternion_Layers as PQL
import pickle


tf.reset_default_graph()

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
Y_train=tf.keras.utils.to_categorical(y_train,10)

lr=tf.placeholder(tf.float32,name='learning_rate')

X=tf.placeholder(tf.float32,shape=(None,28,28))
Y=tf.placeholder(tf.float32,shape=(None,10))


XX=tf.expand_dims(X,3)
with tf.variable_scope('class'):
    w_init=tf.contrib.layers.xavier_initializer()
    b_init=tf.zeros_initializer()
    conv1=tf.nn.conv2d(XX,tf.get_variable('conv1',shape=(5,5,1,120),initializer=w_init),[1,1,1,1],"SAME")
    conv1b=tf.add(conv1,tf.get_variable('bias1',shape=(28,28,120),initializer=b_init))
    A1=tf.nn.relu(conv1b,name='A1')
        
    QA1=tf.reshape(A1,(tf.shape(A1)[0],28*28*120))
        
    QD2=PQL.Pure_Quaternion_Dense_Conjugation(512)(QA1)
        
        
    QA2=PQL.Pure_Quaternion_Mod_Relu(radius=0)(QD2)
        
    QD3=PQL.Pure_Quaternion_Dense_Conjugation(256)(QA2)
        
    QA3=PQL.Pure_Quaternion_Mod_Relu(radius=0)(QD3)
        
    QD4=PQL.Pure_Quaternion_Dense_Conjugation(10)(QA3)
        
    qform=tf.reshape(QD4,(tf.shape(QD4)[0],10,3))
        
    squarenorms=tf.reduce_sum(qform*qform,axis=-1)
        
    O=tf.nn.softmax(tf.sqrt(squarenorms))
    

Loss=tf.reduce_mean((Y-O)**2)
TV=tf.trainable_variables()
Optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(Loss,var_list=TV)
grad=tf.gradients(Loss,TV)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    learning_rate=10
    L=[]
    Points={}
    for epoch in range(10):
        print('\nEpoch: {}'.format(epoch))
        LE=0
        LB=0
        for batch in range(int(x_train.shape[0]/100)):
            _,l,o=sess.run([Optimizer,Loss,O],feed_dict={X:x_train[batch:batch+100,:,:],Y:Y_train[batch:batch+100,:],lr:learning_rate})
            LE+=l
            LB+=l
            if batch%50==0:
                print('Batch {}: {}'.format(batch,LB))
                LB=0
        L.append(LE)
        if epoch>1:
            if learning_rate<1e-8:
                pass
            elif L[-1]>=L[-2]:
                learning_rate*=0.1
                print(learning_rate)
        Points[epoch]=LE
        print('\nTotal Loss: {}'.format(LE))
    with open('H:/dos/GitHub/Pure_10_dic.pkl','wb') as f:
        pickle.dump(Points,f,pickle.HIGHEST_PROTOCOL)


'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    qd2=sess.run(qd2,feed_dict={X:x_train[:100,:,:]})
    print(qd2)
'''
'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    grad=sess.run(grad,feed_dict={X:x_train[:100,:,:],Y:Y_train[:100,:]})
    print(grad)
'''