#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 20-09-2020
File: models.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from myModules.layers import Dense, ConcatInput, Concat, Input
from myModules import losses
from myModules import optimizers
from myModules import metrics



class Network(object):
    def __init__(self,Inputlayer):
        self.layers = np.array([Inputlayer],dtype=object)
        #completar
    
    def add(self,layer):
        # XXX Cambiar la inicializacion para que si len es 0 asuma que el layer que se esta metiendo es correcto
        # import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT
        layer.set_input_shape(self.layers[-1].get_output_shape())
        self.layers = np.append(self.layers, layer)

    def get_layer(self,i):
        return self.layers[i]

    def fit(self,x, y, epochs = 100, bs = 50, loss = losses.MSE(), metric=metrics.acc_XOR,
                    opt = optimizers.SGD(), x_test=None, y_test=None, plot=True):

        self.loss = loss
        self.opt = opt

        loss_v = np.array([])
        acc  = np.array([])
        if (isinstance(x_test, np.ndarray)):
            t_acc = np.array([])
        
        if(plot):
            if(isinstance(x_test, np.ndarray)):
                fig    = plt.figure(figsize=(11,4))
                axloss = plt.subplot(131)
                axacc  = plt.subplot(132)
                axTacc = plt.subplot(133)
            else:
                fig    = plt.figure(figsize=(11,4))
                axloss = plt.subplot(121)
                axacc  = plt.subplot(122)


        for e in range(epochs):

            self.opt(x,y, self, bs=bs)

            loss_v = np.append(loss_v,   loss(self.forward(x) ,y) )
            acc  = np.append(acc  , metric(self.forward(x), y) )
            if(isinstance(x_test, np.ndarray)):
                t_acc  = np.append(t_acc  , metric(self.forward(x_test), y_test) )
            
            if (e % 1 == 0):
                if(isinstance(x_test, np.ndarray)):
                    print("{}/{}\tloss: {:.2f}\tAccurracy: {:.2f}\ttest: {:.2f}".format(
                    e, epochs, loss_v[-1], acc[-1], t_acc[-1] ))
                else:
                    print("{}/{}\tloss: {:.2f}\tAccurracy: {:.2f}".format(e, epochs, loss_v[-1], acc[-1] ) )
            
            if(plot):
                if(isinstance(x_test, np.ndarray)):
                    fig.clf()
                    axloss = plt.subplot(131)
                    axacc  = plt.subplot(132)
                    axTacc = plt.subplot(133)
                    axloss.set_title("Loss"), axacc.set_title("Accurracy"), axTacc.set_title("Test Accurracy")
                    axloss.plot(np.arange(e+1), loss_v)
                    axacc.plot(np.arange(e+1), acc)
                    axTacc.plot(np.arange(e+1), t_acc)
                    plt.tight_layout()
                    plt.pause(0.1)
                else:
                    fig.clf()
                    axloss = plt.subplot(121)
                    axacc  = plt.subplot(122)
                    axloss.set_title("Loss"), axacc.set_title("Accurracy")
                    axloss.plot(np.arange(e+1), loss_v)
                    axacc.plot(np.arange(e+1), acc)
                    plt.tight_layout()
                    plt.pause(0.1)

        
        print("Precision final con los datos de entrenamiento: ", acc[-1])
        if(isinstance(x_test, np.ndarray)):
            print("Precision final con los datos de test: ", t_acc[-1])
        

        plt.close('all')
        plt.ion()

        plt.figure()
        plt.title("Loss")
        plt.plot(np.arange(epochs), loss_v)

        plt.figure()
        plt.title("Accurracy")
        plt.plot(np.arange(epochs), acc)

        if(isinstance(x_test, np.ndarray)):
            plt.figure()
            plt.title("Test Accurracy")
            plt.plot(np.arange(epochs), t_acc)





    # def forward_upto(j = False):
    def forward(self,X,j = None):
        if(j == None):
            j = len(self.layers)
        
        res = np.copy(X)

        for i in range(j):
            if ( isinstance(self.layers[i], Concat) ):
            # if ( isinstance(self.layers[i], ConcatInput) ):
                res = self.layers[i](X,res)
            else:
                res = self.layers[i](res)
        
        return res
        # completar

    def predict(self,X):
        # return np.argmax(self.forward(X), axis=1)
        return self.forward(X)

    def backward(self,X,Y, j=None, grad=None):

        if(j == None):
            j = len(self.layers)
            grad = self.loss.gradient( self.forward(X), Y )

        for i in range(j-1, -1, -1):
        # for i in range(j-1, 0, -1):

            if( isinstance(self.layers[i], Dense) ):

                # import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT

                S = self.forward(X,i)               #XXX Chequear hasta donde lo calculo
                S_i = self.layers[i].dot(S)

                grad = grad * self.layers[i].activation.gradient(S_i)

                S = np.hstack((np.ones((S.shape[0], 1)), S))

                gradW = np.dot(S.T, grad)

                
                # import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT

                grad = np.dot(grad, self.layers[i].get_weights().T )
                grad = grad[:, 1:]

                dW = self.opt.update_weights(self.layers[i], gradW)
                self.layers[i].update_weights(dW)

            elif( isinstance(self.layers[i], Concat) ):
                grad2 = grad[:, self.layers[i].get_input1_shape():]
                grad = grad[:, :self.layers[i].get_input1_shape() ]

                self.backward(X,Y,self.layers[i].i+1 ,grad2)      # Capaz esto es un i+-1

            elif( isinstance(self.layers[i], ConcatInput) ):
                grad = grad[:, :self.layers[i].get_input1_shape() ]