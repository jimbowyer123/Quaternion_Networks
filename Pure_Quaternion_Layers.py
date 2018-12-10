# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:20:54 2018

@author: jb2968
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import Quaternion_Arithmetic as QA

class Quaternion_Dense_Conjugation(Layer):
    
    def __init(self,output_size,**kwargs):
        self.output_dim=output_size
        super(Quaternion_Dense_Conjugation, self).__init__(**kwargs)
        
    def __build__(self,input_shape):
        int_shape=input_shape.as_list()
        assert len(int_shape)==2
        assert int_shape[-1]%3==0
        input_dim=int_shape[-1]//3
        kernel_shape=(input_dim,self.output_dim)
        init=tf.contrib.layers.xavier_initializer()
        self.r_kernel=self.add_weight('R_Kernel',kernel_shape,initializer=init)
        self.i_kernel=self.add_weight('I_Kernel',kernel_shape,initializer=init)
        self.j_kernel=self.add_weight('J_Kernel',kernel_shape,initializer=init)
        self.k_kernel=self.add_weight('K_Kernel',kernel_shape,initializer=init)
        
        
        super(Quaternion_Dense_Conjugation, self).build(input_shape)
        
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