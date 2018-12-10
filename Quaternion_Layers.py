# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 15:28:56 2018

@author: jb2968
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import Quaternion_Arithmetic as QA

class Quaternion_Dense_Layer(Layer):
    
    def __init__(self,output_dim,**kwargs):
        self.output_dim=output_dim
        super(Quaternion_Dense_Layer, self).__init__(**kwargs)
        
    def build(self,input_shape):
        int_shape=input_shape.as_list()
        assert len(int_shape)==2
        assert int_shape[-1]%4==0
        input_dim=int_shape[-1]//4
        kernel_shape=(input_dim,self.output_dim)
        init=tf.contrib.layers.xavier_initializer()
        self.r_kernel=self.add_weight('R_Kernel',kernel_shape,initializer=init)
        self.i_kernel=self.add_weight('I_Kernel',kernel_shape,initializer=init)
        self.j_kernel=self.add_weight('J_Kernel',kernel_shape,initializer=init)
        self.k_kernel=self.add_weight('K_Kernel',kernel_shape,initializer=init)
        
        super(Quaternion_Dense_Layer, self).build(input_shape)
        
        
    def call(self,inputs):
        k_for_r=tf.concat([self.r_kernel, -self.i_kernel, -self.j_kernel, -self.k_kernel],axis=0)
        k_for_i=tf.concat([self.i_kernel, self.r_kernel, -self.k_kernel, self.j_kernel],axis=0)
        k_for_j=tf.concat([self.j_kernel, self.k_kernel, self.r_kernel, -self.i_kernel],axis=0)
        k_for_k=tf.concat([self.k_kernel, -self.j_kernel, self.i_kernel, self.r_kernel],axis=0)
        

        k_for_q=tf.concat([k_for_r,k_for_i,k_for_j,k_for_k],axis=-1)
        
        return K.dot(inputs,k_for_q)

               
        
class Quaternion_Mod_Relu(Layer):
    
    def __init__(self,radius=0.1,**kwargs):
        self.Radius=radius
        super(Quaternion_Mod_Relu, self).__init__(**kwargs)
        
    def __build__(self,input_shape):
        super(Quaternion_Mod_Relu, self).build(input_shape)
        
    def call(self,inputs):
        input_shape=tf.shape(inputs)
        q_inputs=tf.reshape(inputs,(input_shape[0],input_shape[1]//4,4))
        norm=QA.quaternion_norm(q_inputs)
        
        #A=tf.expand_dims(norm,axis=2)
        B=tf.tile(norm,[1,4])
    
        return tf.where(B>self.Radius,inputs,tf.zeros(tf.shape(inputs)))
