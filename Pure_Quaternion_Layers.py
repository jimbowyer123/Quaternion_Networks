# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:20:54 2018

@author: jb2968
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K


class Pure_Quaternion_Dense_Conjugation(Layer):
    
    def __init__(self,output_size,**kwargs):
        self.output_dim=output_size
        print('here')
        super(Pure_Quaternion_Dense_Conjugation, self).__init__(**kwargs)
        
    def build(self,input_shape):
        int_shape=input_shape.as_list()
        assert len(int_shape)==2
        assert int_shape[-1]%3==0
        input_dim=int_shape[-1]//3
        kernel_shape=(input_dim,self.output_dim)
        init=tf.initializers.random_uniform(minval=-0.1,maxval=0.1)
        self.r_kernel=self.add_weight('R_Kernel',kernel_shape,initializer=init)
        self.i_kernel=self.add_weight('I_Kernel',kernel_shape,initializer=init)
        self.j_kernel=self.add_weight('J_Kernel',kernel_shape,initializer=init)
        self.k_kernel=self.add_weight('K_Kernel',kernel_shape,initializer=init)
        
        
        super(Pure_Quaternion_Dense_Conjugation, self).build(input_shape)
        
    def call(self,inputs):
        
        k_for_i=tf.concat([
                tf.square(self.r_kernel)+tf.square(self.i_kernel)-tf.square(self.j_kernel)-tf.square(self.k_kernel),
                2*(self.i_kernel*self.j_kernel-self.r_kernel*self.k_kernel),
                2*(self.i_kernel*self.k_kernel+self.r_kernel*self.j_kernel)
                ]
                ,axis=0)
        
        k_for_j=tf.concat([
                2*(self.r_kernel*self.k_kernel+self.i_kernel*self.j_kernel),
                tf.square(self.r_kernel)-tf.square(self.i_kernel)+tf.square(self.j_kernel)-tf.square(self.k_kernel),
                2*(self.j_kernel*self.k_kernel-self.r_kernel*self.i_kernel)
                ]
                ,axis=0)
        
        k_for_k=tf.concat([
                2*(self.i_kernel*self.k_kernel-self.r_kernel*self.j_kernel),
                2*(self.r_kernel*self.i_kernel+self.j_kernel*self.k_kernel),
                tf.square(self.r_kernel)-tf.square(self.i_kernel)-tf.square(self.j_kernel)+tf.square(self.k_kernel)
                ]
                ,axis=0)
        
        k_for_q=tf.concat([k_for_i,k_for_j,k_for_k],axis=-1)
        
        return K.dot(inputs,k_for_q)
    
class Pure_Quaternion_Mod_Relu(Layer):
    
    def __init__(self,radius=0.1,**kwargs):
        self.Radius=radius
        super(Pure_Quaternion_Mod_Relu,self).__init__(**kwargs)
        
    def build(self,input_shape):
        super(Pure_Quaternion_Mod_Relu, self).build(input_shape)
        
    def call(self,inputs):
        
       quaternion_inputs=tf.reshape(inputs,(tf.shape(inputs)[0],tf.shape(inputs)[1]//3,3))
       norms=tf.sqrt(tf.reduce_sum(quaternion_inputs*quaternion_inputs,axis=-1))
       
       B=tf.tile(norms,[1,3])
       
       return tf.where(B>self.Radius,inputs,tf.zeros(tf.shape(inputs)))