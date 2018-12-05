# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 10:13:25 2018

@author: jb2968
"""

import tensorflow as tf
import numpy as np

def left_tensor_quaternion_mult(P,Q):
    
    QQ=tf.broadcast_to(Q,(P.shape[0],Q.shape[0],Q.shape[1]))
    a=P[:,:,0]*QQ[:,:,0]-tf.reduce_sum(P[:,:,1:]*QQ[:,:,1:],axis=2)
    bcd=tf.reshape(P[:,:,0],(P.shape[0],P.shape[1],1))*QQ[:,:,1:]+ tf.reshape(QQ[:,:,0],(QQ.shape[0],QQ.shape[1],1))*P[:,:,1:] + tf.cross(P[:,:,1:],QQ[:,:,1:])
    
    return tf.concat([tf.reshape(a,(tf.shape(a)[0],tf.shape(a)[1],1)),bcd],axis=2)

def right_tensor_quaternion_mult(P,Q):
    PP=tf.broadcast_to(P,(Q.shape[0],P.shape[0],P.shape[1]))
    a=PP[:,:,0]*Q[:,:,0]-tf.reduce_sum(PP[:,:,1:]*Q[:,:,1:],axis=2)
    bcd=tf.reshape(PP[:,:,0],(PP.shape[0],PP.shape[1],1))*Q[:,:,1:]+ tf.reshape(Q[:,:,0],(Q.shape[0],Q.shape[1],1))*PP[:,:,1:] + tf.cross(PP[:,:,1:],Q[:,:,1:])
    
    return tf.concat([tf.reshape(a,(tf.shape(a)[0],tf.shape(a)[1],1)),bcd],axis=2)

def weight_conjugate(P):
    return tf.concat([tf.expand_dims(P[:,0],axis=1),-1*P[:,1:]],axis=1)

def conjugation(P,Q):
    x=left_tensor_quaternion_mult(P,weight_conjugate(Q))
    y=right_tensor_quaternion_mult(Q,x)
    return y

def tensor_quaternion_addition(tensor,bias):
    BB=tf.broadcast_to(bias,(tensor.shape[0],bias.shape[0],bias.shape[1]))
    
    return tensor+BB

def quaternion_fully_connected(tensor,weights,bias):
    '''
    tensor of shape (batch_size,input_layer_size,4)
    weights of shape (output_layer_size,input_layer_size,4)
    bias of shape (output_layer_size,4)
    '''
    
    
    WW=tf.tile(tf.expand_dims(weights,axis=0),[tf.shape(tensor)[0],1,1,1])
    BB=tf.tile(tf.expand_dims(bias,axis=0),[tf.shape(tensor)[0],1,1])
    TT=tensor
    
    print(WW)
    print(TT)
    
    #WWW1,WWW2,WWW3,WWW4=tf.split(WW,4,axis=-1)
    TT1,TT2,TT3,TT4=tf.split(TT,4,axis=-1)
    

    
    WW1=WW[:,:,:,0]
    WW2=WW[:,:,:,1]
    WW3=WW[:,:,:,2]
    WW4=WW[:,:,:,3]
    
    print(WW1)
    print(TT1)
    
    A=tf.expand_dims(tf.matmul(WW1,TT1)-tf.matmul(WW2,TT2)-tf.matmul(WW3,TT3)-tf.matmul(WW4,TT4),axis=2)
    B=tf.expand_dims(tf.matmul(WW3,TT4)-tf.matmul(WW4,TT3)+tf.matmul(WW1,TT2)+tf.matmul(WW2,TT1),axis=2)
    C=tf.expand_dims(tf.matmul(WW4,TT2)-tf.matmul(WW2,TT4)+tf.matmul(WW1,TT3)+tf.matmul(WW3,TT1),axis=2)
    D=tf.expand_dims(tf.matmul(WW2,TT3)-tf.matmul(WW3,TT2)+tf.matmul(WW1,TT4)+tf.matmul(WW4,TT1),axis=2)
    
    WWTT=tf.concat([A,B,C,D],axis=2)
    Z=tf.squeeze(WWTT)+BB
    
    return Z

def quaternion_norm(quaternion_tensor):
    squares=quaternion_tensor*quaternion_tensor
    
    return tf.reduce_sum(squares,axis=-1)**0.5

def quaternion_mod_relu(quaternion_tensor,radius):
    
    return tf.where(quaternion_norm(quaternion_tensor)>radius,quaternion_tensor,tf.zeros(tf.shape(quaternion_tensor)))
    
    
    
    
    