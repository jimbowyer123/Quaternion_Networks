# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 11:08:33 2018

@author: jb2968
"""

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

X=tf.placeholder(tf.float32,shape=(None,28,28),name='X')
Y=tf.placeholder(tf.float32,shape=(None,10),name='Y')


XX=tf.expand_dims(X,3)

w_init=tf.contrib.layers.xavier_initializer()
b_init=tf.zeros_initializer()

with tf.variable_scope('Conv1'):
    conv1=tf.nn.conv2d(XX,tf.get_variable('weights',shape=(5,5,1,120),initializer=w_init),[1,1,1,1],"SAME")

    conv1b=tf.add(conv1,tf.get_variable('bias',shape=(28,28,120),initializer=b_init))

    A1=tf.nn.relu(conv1b,name='Relu')    

QA1=tf.reshape(A1,(tf.shape(A1)[0],28*28*120))
    
with tf.variable_scope('Dense2'):
    QD2=PQL.Pure_Quaternion_Dense_Conjugation(512)(QA1)
        
    QA2=PQL.Pure_Quaternion_Mod_Relu(radius=0)(QD2)
    
with tf.variable_scope('Dense3'):
        
    QD3=PQL.Pure_Quaternion_Dense_Conjugation(256)(QA2)
        
    QA3=PQL.Pure_Quaternion_Mod_Relu(radius=0)(QD3)
        
with tf.variable_scope('Dense4'):

    QD4=PQL.Pure_Quaternion_Dense_Conjugation(10)(QA3)
    
qform=tf.reshape(QD4,(tf.shape(QD4)[0],10,3))
        
squarenorms=tf.reduce_sum(qform*qform,axis=-1,name='Square_Norms')
    
O=tf.nn.softmax(tf.sqrt(squarenorms))
    

Loss=tf.reduce_mean((Y-O)**2,name='Loss')
TV=tf.trainable_variables()
Optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(Loss,var_list=TV,name='Train')
#grad=tf.gradients(Loss,TV)
tf.summary.scalar('loss',Loss)
merged_summary=tf.summary.merge_all()

with tf.Session() as sess:
    writer=tf.summary.FileWriter('C:/Users/jb2968/Tensor_Board/Quaternion_Networks/Pure_Quaternion_Network')
    writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    learning_rate=10
    progbar=tf.keras.utils.Progbar(int(x_train.shape[0]/100))
    for batch in range(int(x_train.shape[0]/100)):
        progbar.update(batch)
        _=sess.run(Optimizer,feed_dict={X:x_train[batch:batch+100,:,:],Y:Y_train[batch:batch+100,:],lr:learning_rate})
        
        if batch%50==0:
            ms=sess.run(merged_summary,feed_dict={X:x_train[batch:batch+100,:,:],Y:Y_train[batch:batch+100,:]})
            writer.add_summary(ms,batch)


'''
with tf.Session() as sess:
    writer=tf.summary.FileWriter('C:/Users/jb2968/Tensor_Board/2')
    writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    learning_rate=10
    L=[]
    Points={}
    for epoch in range(1):
        print('\nEpoch: {}'.format(epoch))
        LE=0
        LB=0
        for batch in range(int(x_train.shape[0]/100)):
            _,l,o,ms=sess.run([Optimizer,Loss,O,merged_summary],feed_dict={X:x_train[batch:batch+100,:,:],Y:Y_train[batch:batch+100,:],lr:learning_rate})
            writer.add_summary(ms)
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
    with open('H:/dos/GitHub/Quaternion_Networks/Pure_10_dic.pkl','wb') as f:
        pickle.dump(Points,f,pickle.HIGHEST_PROTOCOL)
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
    
    grad=sess.run(grad,feed_dict={X:x_train[:100,:,:],Y:Y_train[:100,:]})
    print(grad)
'''