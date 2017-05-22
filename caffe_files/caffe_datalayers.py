#!/usr/bin/env python

import os
import numpy as np
import numpy.matlib as ml
import sys
import glob
import scipy
import scipy.linalg as scl
import scipy.io as scio
import matplotlib.pyplot as plt
import datetime
import time
# import img_xforms
# import imgnet_xform
import itertools
import caffe
import rz_fcns_nohdf5 as rz
import h5py
import color_quantization as cq
from skimage.transform import pyramid_gaussian as pg
import warnings

# *****************************************************
# ******************** LAYERS USED ********************
# *****************************************************
class NNEncLayer(caffe.Layer):
    ''' Layer which encodes ab map into Q colors
    INPUTS    
        bottom[0]   Nx2xXxY     
    OUTPUTS
        top[0].data     NxQ     
    '''
    def setup(self,bottom, top):
        warnings.filterwarnings("ignore")

        if len(bottom) == 0:
            raise Exception("Layer should have inputs")
        # self.NN = 10.
        self.NN = 1.
        self.sigma = 5.
        self.ENC_DIR = './data/color_bins'
        self.nnenc = cq.NNEncode(self.NN,self.sigma,km_filepath=os.path.join(self.ENC_DIR,'pts_in_hull.npy'))

        self.N = bottom[0].data.shape[0]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]
        self.Q = self.nnenc.K

    def reshape(self, bottom, top):
        top[0].reshape(self.N,self.Q,self.X,self.Y)
 
    def forward(self, bottom, top):
        top[0].data[...] = self.nnenc.encode_points_mtx_nd(bottom[0].data[...],axis=1)

    def backward(self, top, propagate_down, bottom):
        # no back-prop
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)

class PriorBoostLayer(caffe.Layer):
    ''' Layer boosts ab values based on their rarity
    INPUTS    
        bottom[0]       NxQxXxY     
    OUTPUTS
        top[0].data     Nx1xXxY
    '''
    def setup(self,bottom, top):
        if len(bottom) == 0:
            raise Exception("Layer should have inputs")

        self.ENC_DIR = './data/color_bins'
        self.gamma = .5
        self.alpha = 1.
        self.pc = PriorFactor(self.alpha,gamma=self.gamma,priorFile=os.path.join(self.ENC_DIR,'prior_probs.npy'))

        self.N = bottom[0].data.shape[0]
        self.Q = bottom[0].data.shape[1]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]

    def reshape(self, bottom, top):
        top[0].reshape(self.N,1,self.X,self.Y)
 
    def forward(self, bottom, top):
        top[0].data[...] = self.pc.forward(bottom[0].data[...],axis=1)

    def backward(self, top, propagate_down, bottom):
        # no back-prop
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)

class NonGrayMaskLayer(caffe.Layer):
    ''' Layer 
    INPUTS    
        bottom[0]       Nx2xXxY     ab values
    OUTPUTS
        top[0].data     Nx1xXxY     1 if image is NOT grayscale
                                    0 if image is grayscale
    '''
    def setup(self,bottom, top):
        if len(bottom) == 0:
            raise Exception("Layer should have inputs")

        self.N = bottom[0].data.shape[0]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]
        self.thresh = 5

    def reshape(self, bottom, top):
        top[0].reshape(self.N,1,self.X,self.Y)
 
    def forward(self, bottom, top):
        top[0].data[...] = (np.sum(np.sum(np.sum(np.abs(bottom[0].data) > self.thresh,axis=1),axis=1),axis=1) > 0)[:,rz.na(),rz.na(),rz.na()]

    def backward(self, top, propagate_down, bottom):
        # no back-prop
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)


class PairOverlapLayer(caffe.Layer):
    ''' Layer which computes cosine similarity on random indices for all inputs
    Same indices are used for all of the features from the bottom
    INPUTS
      bottom[0]       NxQxXxY     probability distribution
      bottom[1]       NxQxXxY     probability distribution
      ...
    OUTPUTS
      top[0].data     NxP         distribution overlaps
      top[1].data     NxP         distribution overlaps
      ...   '''
    def setup(self, bottom, top):
        if len(bottom) == 0:
            raise Exception("Layer should have inputs")
        self.P = 121
        self.pos = {}
        for ii in range(len(bottom)):
            self.pos[ii] = PairOverlap(bottom[0].data,self.P)
        (self.N,self.Q,self.X,self.Y) = bottom[0].shape
        self.gen_rnd_indices()

    def gen_rnd_indices(self):
        # generate random indices, propogate to all pairoverlap objects
        self.pos[0].gen_rnd_inds()
        for ii in np.arange(1,len(self.pos)):
            self.pos[ii].set_inds(self.pos[0].inds0,self.pos[0].inds1)

    def reshape(self, bottom, top):
        for ii in range(len(bottom)):
            top[ii].reshape(self.N,self.P)
        # top[1].reshape(self.N,self.P)
 
    def forward(self, bottom, top):
        self.gen_rnd_indices() # generate random indices
        for ii in np.arange(0,len(bottom)):
            top[ii].data[...] = self.pos[ii].forward(bottom[ii].data)

        # plt.figure()
        # plt.plot(np.concatenate((self.pos[0].inds0[:,[0]],self.pos[0].inds1[:,[0]]),axis=1),
        #     np.concatenate((self.pos[0].inds0[:,[1]],self.pos[0].inds1[:,[1]]),axis=1),'go-')
        # # plt.plot(self.pos[0].inds1[:,0],self.pos[0].inds1[:,1],'ro')

        # plt.figure()
        # plt.hist(np.linalg.norm(self.pos[0].inds0-self.pos[0].inds1,axis=1),bins=np.arange(0,100,1))
        # plt.show()

    def backward(self, top, propagate_down, bottom):
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = self.pos[i].backward(top[i].diff[...])

class SubsampleIndicesLayer(caffe.Layer):
    ''' Layer which routes random subsample of indices upwards.
    INPUTS
      bottom[ii].data  NxQiixXxY     probability distribution
    OUTPUTS
      top[ii].data     NxQiixPx1     randomly subsampled points routed up '''
    def setup(self, bottom, top):
        if len(bottom) == 0:
            raise Exception("Data layer should have inputs")
        self.P = 242 # number of points
        self.ssl = {} # subsample index objects
        self.Qs = np.zeros(len(bottom),dtype='int')
        for ii in range(len(bottom)):
            self.ssl[ii] = SubsampleIndices(bottom[ii].data,self.P)
            (self.N,self.Qs[ii],self.X,self.Y) = bottom[ii].shape
        self.gen_rnd_indices()

    def gen_rnd_indices(self):
        ''' generate random indices, propagate to all subsample index objects '''
        self.ssl[0].gen_rnd_inds() # generate random indices
        for ii in np.arange(1,len(self.ssl)): # propagate to other objects
            self.ssl[ii].set_inds(self.ssl[0].inds)

    def reshape(self, bottom, top):
        for ii in range(len(bottom)):
            top[ii].reshape(self.N,self.Qs[ii],self.P,1)
 
    def forward(self, bottom, top):
        self.gen_rnd_indices() # generate random indices
        for ii in np.arange(0,len(bottom)):
            top[ii].data[:,:,:,0] = self.ssl[ii].forward(bottom[ii].data)

    def backward(self, top, propagate_down, bottom):
        # print 'Back-propagating subsampling indices layer'
        # print self.Qs
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            # print 'Back-propagating subsample indices layer: %i'%i
            bottom[i].diff[:,:,:,:] = self.ssl[i].backward(top[i].diff[:,:,:,0])

class TemperatureLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.temp = 10
        self.verbose = False

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].shape[0],)

    def forward(self, bottom, top):
        top[0].data[...] = self.temp
        if(self.verbose):
            print(self.temp)

    def backward(self, top, propagate_down, bottom):
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = 0

class Temperature20Layer(TemperatureLayer):
    def setup(self, bottom, top):
        self.temp = 250
        self.verbose = False


# ***********************************************************
# ******************** CAFFE DATA LAYERS ********************
# ***********************************************************
class ILSVRCTrnLabDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 0:
            raise Exception("Data layer should not have inputs")
        self.verbose = True
        self.X = 176
        self.N = 40
        self.Y = self.X
        self.B = 129 # total number of files
        self.TO_PROD_SAME = False
        self.SS = 4
        self.XSS = self.X/self.SS
        self.YSS = self.Y/self.SS
        self.sigma = 5 # sigma

        self.ENC_DIR = './data/color_bins'
        self.ildl = ILSVRCLabDataLoader(B=self.B,N=self.N,sameMode=self.TO_PROD_SAME, fileprefix='trn_data_lab_randord_224',verbose=self.verbose,filedir='/data/efros/rzhang/datasets/ILSVRC2012_processed')
        self.data_empty = True

    def reshape(self, bottom, top):
        top[0].reshape(self.N,1,self.X,self.Y)
        top[1].reshape(self.N,2,self.XSS,self.YSS)
        # top[1].reshape(self.N,2,self.X,self.Y)
 
    def forward(self, bottom, top):
        # self.alpha = 1. # prior probability correction factor
        # self.gamma = .5 # mix in uniform prior
        # self.pc = PriorFactor(self.alpha,gamma=self.gamma,verbose=True,priorFile=os.path.join(self.ENC_DIR,'prior_probs.npy'))
        # self.TO_PROD_SAME = False

        # self.pc.print_correction_stats()
        randJitter = True
        randFlip = True
        # if(self.data_empty or not self.TO_PROD_SAME):
            # self.data_empty = False
        # print 'Passing forward'
        (data_l,data_ab,grayMask) = self.ildl.forward(self.X,self.Y,SS=self.SS,returnGrayMask=True,randJitter=randJitter,randFlip=randFlip)
        top[0].data[:,:,:,:] = data_l
        top[1].data[:,:,:,:] = data_ab
        # top[2].data[grayMask,:,:,:] = 0

    def backward(self, top, propagate_down, bottom):
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)

# OUTPUTS
#   data_l                  Nx1xXxY         
#   data_ab_quant_map_233   NxQxXssxYss     quantized
#   data_ab_prior_boost     Nx1xXssxYss     boost factor, based on prior
class ILSVRCTrnLabQuantNNPriorBoostDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 0:
            raise Exception("Data layer should not have inputs")
        self.verbose = True
        # self.X = 224
        # self.N = 68 # batch size
        # self.X = 200
        # self.N = 84 # batch size
        self.X = 176
        # self.N = 100 # batch size
        # self.N = 45
        self.N = 40
        # self.N = 20
        self.Y = self.X
        self.B = 129 # total number of files
        # self.TO_PROD_SAME = True # keep spitting out the first minibatch
        self.TO_PROD_SAME = False
        self.NN = 10 # nearest neighbors
        # self.SS = 8
        self.SS = 4
        self.XSS = self.X/self.SS
        self.YSS = self.Y/self.SS
        self.sigma = 5 # sigma
        # self.alpha = 0.20 # prior probability correction factor
        # self.gamma = 0
        # self.per = 0.5

        # ***** Settings for doing no prior boosting *****
        # self.gamma = 1. # assume uniform prior to begin with

        # ***** Arithemtic averaging *****
        # self.gamma = .9 # mix in uniform prior
        # self.gamma = .75 # mix in uniform prior
        self.gamma = .5 # mix in uniform prior
        # self.gamma = .25 # mix in uniform prior

        # self.alpha = 0. # prior probability correction factor
        self.alpha = 1. # prior probability correction factor

        self.ENC_DIR = './data/color_bins'
        self.ildl = ILSVRCLabDataLoader(B=self.B,N=self.N,sameMode=self.TO_PROD_SAME, fileprefix='trn_data_lab_randord_224',verbose=self.verbose,filedir='/data/efros/rzhang/datasets/ILSVRC2012_processed')
        self.nnenc = cq.NNEncode(self.NN,self.sigma,km_filepath=os.path.join(self.ENC_DIR,'pts_in_hull.npy'))
        self.pc = PriorFactor(self.alpha,gamma=self.gamma,priorFile=os.path.join(self.ENC_DIR,'prior_probs.npy'))
        # self.ilpl = ILSVRCLabPaletteLoader(B=self.B,N=self.N,sameMode=self.TO_PROD_SAME, fileprefix='trn_data_randord_palette',verbose=self.verbose,filedir='/data/efros/rzhang/datasets/ILSVRC2012_processed')

        self.data_empty = True

    def reshape(self, bottom, top):
        top[0].reshape(self.N,1,self.X,self.Y)
        top[1].reshape(self.N,self.nnenc.K,self.XSS,self.YSS)
        top[2].reshape(self.N,1,self.XSS,self.YSS)
 
    def forward(self, bottom, top):
        # self.alpha = 1. # prior probability correction factor
        # self.gamma = .5 # mix in uniform prior
        # self.pc = PriorFactor(self.alpha,gamma=self.gamma,verbose=True,priorFile=os.path.join(self.ENC_DIR,'prior_probs.npy'))
        # self.TO_PROD_SAME = False

        # self.pc.print_correction_stats()
        randJitter = True
        randFlip = True
        # if(self.data_empty or not self.TO_PROD_SAME):
            # self.data_empty = False
        # print 'Passing forward'
        (data_l,data_ab,grayMask) = self.ildl.forward(self.X,self.Y,SS=self.SS,returnGrayMask=True,randJitter=randJitter,randFlip=randFlip)
        top[0].data[...] = data_l
        top[1].data[...] = self.nnenc.encode_points_mtx_nd(data_ab,axis=1)
        top[2].data[...] = self.pc.forward(top[1].data[...],axis=1)
        top[2].data[grayMask,:,:,:] = 0

        # img_xform = rz.img_data_lab_transformer()
        # plt.figure()
        # plt.imshow(rz.montage(img_xform.data2img(np.concatenate((data_l[:,:,::self.SS,::self.SS],data_ab),axis=1))))
        # plt.show()

        # if(np.sum(grayMask)):
            # print '  Images b&w: %i'%np.sum(grayMask)

    def backward(self, top, propagate_down, bottom):
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)



# *****************************************************
# ******************** DATA LAYERS ********************
# *****************************************************
class PairOverlap():
    ''' Given a blob of probabilities, compute histogram cosine similarity for
    pairs of locations. Locations can be randomly generated. '''
    def __init__(self,bottom_data,P):
        self.P = P
        (self.N,self.Q,self.X,self.Y) = bottom_data.shape
        self.gen_rnd_inds()

    def gen_rnd_inds(self,method=2,bord=1.,scale=2.):
        ''' Generate pairs of indices
        INPUTS
            method          integer         [2]-Gaussian, 1-Laplacian, 0-uniform, 3-random, 4-same indices
            bord            integer         number of border pixels to leave for first index
            scale           float           scale to draw random data from
        SAVED
            inds0,inds1     Nx2             sampled points '''

        # Generate new set of P indices
        self.inds0 = np.zeros((self.P,2),dtype='int')
        self.inds0[:,0] = bord + np.random.randint(self.X-2*bord,size=(self.P,))
        self.inds0[:,1] = bord + np.random.randint(self.Y-2*bord,size=(self.P,))

        if(method==0):
            dinds = np.round(np.random.randint(scale,size=(self.P,2))).astype('int')
            self.inds1 = self.inds0 + dinds
        elif(method==1):
            dinds = np.round(np.random.laplace(scale=scale,size=(self.P,2))).astype('int')
            self.inds1 = self.inds0 + dinds
        elif(method==2):
            dinds = np.round(np.random.normal(scale=scale,size=(self.P,2))).astype('int')
            self.inds1 = self.inds0 + dinds
        elif(method==3):
            self.inds1 = np.zeros((self.P,2),dtype='int') # Completely random
            self.inds1[:,0] = np.random.randint(self.X,size=(self.P,))
            self.inds1[:,1] = np.random.randint(self.Y,size=(self.P,))
        elif(method==4):
            self.inds1 = self.inds0.copy() # Same indices

        # clip indices
        self.inds1[:,0] = np.clip(self.inds1[:,0],0,self.X-1)
        self.inds1[:,1] = np.clip(self.inds1[:,1],0,self.Y-1)

    def set_inds(self,inds0,inds1):
        ''' Set indices
        SAVED
          inds0       Px2           
          inds1       Px2   '''
        self.inds0 = inds0.copy()
        self.inds1 = inds1.copy()

    def forward(self,bottom_data):
        '''
        INPUTS
          bottom_mtx  NxQxXxY       probability distributions
        SAVED
          qs0         NxQxP         probability distribution at inds0
          qs1         NxQxP         probability disribution at inds1 
          q_overlap   NxP           cosine similarity scores
        OUTPUT
          q_overlap   NxP           cosine similarity scores       '''

        self.qs0 = bottom_data[:,:,self.inds0[:,0],self.inds0[:,1]]
        self.qs1 = bottom_data[:,:,self.inds1[:,0],self.inds1[:,1]]
        self.q_overlap = np.sum((self.qs0*self.qs1),axis=1)
        return self.q_overlap

    def backward(self,top_diff=1):
        ''' Assume we have just run forward
        INPUTS
          top_diff        NxP       diff signal from top
        OUTPUTS
          bottom_diff     NxQxXxY   diff signal going to bottom
        bottom_diff[ii] = top_diff * qs[1-ii]        '''

        # initialize to zero
        bottom_diff = np.zeros((self.N,self.Q,self.X,self.Y),dtype='float32')

        # index into qs0, add values of top_diff*qs1
        # index into qs1, add values of top_diff*qs0
        bottom_diff[:,:,self.inds0[:,0],self.inds0[:,1]] = top_diff[:,rz.na(),:]*self.qs1 + bottom_diff[:,:,self.inds0[:,0],self.inds0[:,1]]
        bottom_diff[:,:,self.inds1[:,0],self.inds1[:,1]] = top_diff[:,rz.na(),:]*self.qs0 + bottom_diff[:,:,self.inds1[:,0],self.inds1[:,1]]

        return bottom_diff

class SubsampleIndices():
    ''' Given a 4D blob subsample random spatial locations '''
    def __init__(self,bottom_data,P):
        ''' Initialize
        INPUTS
            bottom_data     NxQxXxY     an example bottom blob
            P               integer     number of indices
        SAVED
            N,Q,X,Y         integers    blob dimensions '''
        self.P = P
        (self.N,self.Q,self.X,self.Y) = bottom_data.shape
        self.gen_rnd_inds()

    def gen_rnd_inds(self):
        ''' Generate pairs of indices
        SAVED
            inds           Px2          generated indices '''

        # Generate new set of P indices
        self.inds = np.zeros((self.P,2),dtype='int')
        self.inds[:,0] = np.random.randint(self.X,size=(self.P,))
        self.inds[:,1] = np.random.randint(self.Y,size=(self.P,))

    def set_inds(self,inds):
        ''' Set indices
        SAVED
          inds          Px2   '''
        self.inds = inds.copy()

    def forward(self,bottom_data):
        '''
        INPUTS
          bottom_mtx    NxQxXxY       probability distributions
        OUTPUTS
          q             NxQxP         '''

        self.q = bottom_data[:,:,self.inds[:,0],self.inds[:,1]]
        return self.q

    def backward(self,top_diff=1):
        ''' Assume we have just run forward
        INPUTS
          top_diff        NxQxP     diff signal from top
        OUTPUTS
          bottom_diff     NxQxXxY   diff signal going to bottom
        index into bottom indices and pass gradients downwards        '''

        # initialize to zero
        bottom_diff = np.zeros((self.N,self.Q,self.X,self.Y),dtype='float32')
        bottom_diff[:,:,self.inds[:,0],self.inds[:,1]] = top_diff[...]
        return bottom_diff

class PriorFactor():
    def __init__(self,alpha,gamma=0,verbose=True,priorFile='',genc=-1):
        # INPUTS
        #   alpha           integer     prior correction factor, 0 to ignore prior, 1 to divide by prior, alpha to divide by prior^alpha power
        #   gamma           integer     percentage to mix in prior probability
        #   per             [0,1.]      percentile to normalize to, 0 means min, 1 means max
        #   priorFile       file        file which contains prior probabilities across classes    

        self.alpha = alpha
        self.prior_probs = np.load(priorFile)
        self.verbose = verbose

        # mix in uniform spacing
        self.uni_probs = np.zeros_like(self.prior_probs)
        self.uni_probs[self.prior_probs!=0] = 1.
        self.uni_probs = self.uni_probs/np.sum(self.uni_probs)
       
        self.gamma = gamma
        self.prior_mix = (1-self.gamma)*self.prior_probs + self.gamma*self.uni_probs

        # set prior factor
        self.prior_factor = self.prior_mix**-self.alpha
        self.prior_factor = self.prior_factor/np.sum(self.prior_probs*self.prior_factor) # normalize

        self.implied_prior = self.prior_probs*self.prior_factor
        self.implied_prior = self.implied_prior/np.sum(self.implied_prior)

        # add this to the softmax score
        self.softmax_correction = np.log(self.prior_probs/self.implied_prior * (1-self.implied_prior)/(1-self.prior_probs))

        if(self.verbose):
            self.print_correction_stats()

        if(not rz.check_value(genc,-1)):
            self.expand_grid(genc)

    def expand_grid(self,genc):
        self.prior_probs_full_grid = genc.enc_full_grid_mtx_nd(self.prior_probs,axis=0,returnGrid=True)
        self.uni_probs_full_grid = genc.enc_full_grid_mtx_nd(self.uni_probs,axis=0,returnGrid=True)
        self.prior_mix_full_grid = genc.enc_full_grid_mtx_nd(self.prior_mix,axis=0,returnGrid=True)
        self.prior_factor_full_grid = genc.enc_full_grid_mtx_nd(self.prior_factor,axis=0,returnGrid=True)
        self.implied_prior_full_grid = genc.enc_full_grid_mtx_nd(self.implied_prior,axis=0,returnGrid=True)
        self.softmax_correction_full_grid = genc.enc_full_grid_mtx_nd(self.softmax_correction,axis=0,returnGrid=True)

    def print_correction_stats(self):
        print('Prior factor correction:')
        print('  (alpha,gamma) = (%.2f, %.2f)'%(self.alpha,self.gamma))
        print('  (min,max,mean,med,exp) = (%.2f, %.2f, %.2f, %.2f, %.2f)'%(np.min(self.prior_factor),np.max(self.prior_factor),np.mean(self.prior_factor),np.median(self.prior_factor),np.sum(self.prior_factor*self.prior_probs)))

    def forward(self,data_ab_quant,axis=1):
        # data_ab_quant = net.blobs['data_ab_quant_map_233'].data[...]
        data_ab_maxind = np.argmax(data_ab_quant,axis=axis)
        corr_factor = self.prior_factor[data_ab_maxind]
        if(axis==0):
            return corr_factor[rz.na(),:]
        elif(axis==1):
            return corr_factor[:,rz.na(),:]
        elif(axis==2):
            return corr_factor[:,:,rz.na(),:]
        elif(axis==3):
            return corr_factor[:,:,:,rz.na()]

# **********************************************
# *************** Repacking layer **************
# **********************************************
class Unpack2ForwardLayer(caffe.Layer):
    ''' (N,C,X,Y) ==> (N*SS^2,C,X/SS,Y/SS) '''
    def setup(self, bottom, top):
        # check input pair
        if len(bottom)==0:
            raise Exception("Specify inputs")
        self.SS = 2

    def reshape(self, bottom, top):
        (self.N,self.C,self.X,self.Y) = bottom[0].data.shape
        self.Xss = self.X/self.SS
        self.Yss = self.Y/self.SS
        # print bottom[0].data.shape
        # print self.N
        # print self.SS
        # print self.C
        # print self.Xss
        # print self.Yss
        # print self.N*self.SS**2
        top[0].reshape(self.N*self.SS*self.SS,self.C,self.Xss,self.Yss)

    def forward(self, bottom, top):
        out_top = np.zeros((self.N,self.SS,self.SS,self.C,self.Xss,self.Yss),dtype='float32')
        for ss0 in np.arange(self.SS):
            for ss1 in np.arange(self.SS):
                out_top[:,ss0,ss1,:,:,:] = bottom[0].data[:,:,ss0::self.SS,ss1::self.SS]
        top[0].data[...] = out_top.reshape((self.N*self.SS*self.SS,self.C,self.Xss,self.Yss))

    def backward(self, top, propagate_down, bottom): # not implemented
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[ii].diff[...] = 0
            # bottom[0].diff[...] = top[0].diff[...]*bottom[1].data[...]
            # print 'Back-propagating class rebalance, %i'%i

class Repack2ForwardLayer(caffe.Layer):
    ''' (N*SS^2,C,X/SS,Y/SS) ==> (N,C,X,Y) '''
    def setup(self, bottom, top):
        # check input pair
        if len(bottom)==0:
            raise Exception("Specify inputs")
        self.SS = 2

    def reshape(self, bottom, top):
        (self.NSS2,self.C,self.Xss,self.Yss) = bottom[0].data.shape
        self.N = self.NSS2 / (self.SS**2)
        self.X = self.Xss*self.SS
        self.Y = self.Yss*self.SS
        # top[0].reshape((self.N*self.SS*self.SS,self.C,self.Xss,self.Yss))
        top[0].reshape(self.N,self.C,self.X,self.Y)

    def forward(self, bottom, top):
        in_bottom = bottom[0].data.reshape((self.N,self.SS,self.SS,self.C,self.Xss,self.Yss))
        for ss0 in np.arange(self.SS):
            for ss1 in np.arange(self.SS):
                top[0].data[:,:,ss0::self.SS,ss1::self.SS] = in_bottom[:,ss0,ss1,:,:,:]

    def backward(self, top, propagate_down, bottom): # not implemented
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[ii].diff[...] = 0

# ******************************************************
# *************** Class Rebalancing Layer **************
# ******************************************************
class ClassRebalanceMultLayer(caffe.Layer):
# '''
# INPUTS
#     bottom[0]   NxMxXxY     feature map
#     bottom[1]   Nx1xXxY     boost coefficients
# OUTPUTS
#     top[0]      NxMxXxY     on forward, gets copied from bottom[0]
# FUNCTIONALITY
#     On forward pass, top[0] passes bottom[0]
#     On backward pass, bottom[0] gets boosted by bottom[1]
#     through pointwise multiplication (with singleton expansion) '''
    def setup(self, bottom, top):
        # check input pair
        if len(bottom)==0:
            raise Exception("Specify inputs")

    def reshape(self, bottom, top):
        i = 0
        if(bottom[i].data.ndim==1):
            top[i].reshape(bottom[i].data.shape[0])
        elif(bottom[i].data.ndim==2):
            top[i].reshape(bottom[i].data.shape[0], bottom[i].data.shape[1])
        elif(bottom[i].data.ndim==4):
            top[i].reshape(bottom[i].data.shape[0], bottom[i].data.shape[1], bottom[i].data.shape[2], bottom[i].data.shape[3])

    def forward(self, bottom, top):
        # output equation to negative of inputs
        top[0].data[...] = bottom[0].data[...]
        # top[0].data[...] = bottom[0].data[...]*bottom[1].data[...] # this was bad, would mess up the gradients going up

    def backward(self, top, propagate_down, bottom):
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[0].diff[...] = top[0].diff[...]*bottom[1].data[...]
            # print 'Back-propagating class rebalance, %i'%i

# ***********************************************************
# *************** GARBAGE/NOT CURRENTLY USING ***************
# ***********************************************************
# Method loads images along with their corresponding palettes.
# Then quantize the palettes.
class ILSVRCTrnLabPaletteQuantNNDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 0:
            raise Exception("Data layer should not have inputs")
        self.verbose = True
        self.X = 224
        self.Y = self.X
        self.N = 60
        # self.B = 129 # total number of files
        self.B = 1
        self.SS = self.N # subsample on encoded ab output
        self.verbose = True
        # self.TO_PROD_SAME = True # keep spitting out the first minibatch
        self.TO_PROD_SAME = False # keep spitting out the first minibatch
        self.ildl = ILSVRCLabDataLoader(B=self.B,N=self.N,sameMode=self.TO_PROD_SAME, fileprefix='trn_data_lab_randord_224',verbose=self.verbose,filedir='/data/efros/rzhang/datasets/ILSVRC2012_processed')
        self.ilpl = ILSVRCLabPaletteLoader(B=self.B,N=self.N,sameMode=self.TO_PROD_SAME, fileprefix='trn_data_randord_palette',verbose=self.verbose,filedir='/data/efros/rzhang/datasets/ILSVRC2012_processed')

        # Encoder
        self.NN = 10 # nearest neighbors
        self.sigma = 5 # sigma
        self.genc = cq.GridEncode(self.NN,self.sigma)
        self.AB = self.genc.AB_hull

    def reshape(self, bottom, top):
        top[0].reshape(self.N,1,self.X,self.Y)
        top[1].reshape(self.N,self.genc.AB_hull,1,1)
 
    def forward(self, bottom, top):
        cur_db = self.ildl.cur_db
        cur_cnt = self.ildl.cur_cnt

        data_l = self.ildl.forward(self.X,self.Y,SS=self.SS,returnAB=False,override_db=cur_db,override_cnt=cur_cnt)
        palette_k_ab = self.ilpl.forward(override_db=cur_db,override_cnt=cur_cnt).transpose((0,2,1))
        palette_k_quant = self.genc.encode_nn_mtx_nd(palette_k_ab,axis=1)
        palette_quant = np.max(palette_k_quant,axis=2)
        
        # print self.NN
        top[0].data[:,:,:,:] = data_l
        top[1].data[:,:,0,0] = palette_quant

    def backward(self, top, propagate_down, bottom):
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)

# OUTPUTS
#   data_l                  Nx1xXxY         
#   data_ab_quant_2         NxQxX2xY2       quantized color, quarter resolution
#   data_ab_pboost_2        Nx1xX2xY2       boost factor, quarter resolution
#   data_ab_quant_3         NxQxX3xY3       quantized color, eighth resolution
#   data_ab_pboost_3        Nx1xX3xY3       boost factor, eighth resolution
#   data_ab_quant_4         NxQxX4xY4       quantized color, sixteenth resolution
#   data_ab_pboost_4        Nx1xX4xY4       boost factor, sixteenth resolution
#   data_ab_quant_0         NxQxX0xY0       quantized color, full resolution
#   data_ab_pboost_0        Nx1xX0xY0       boost factor, full resolution
#   data_ab_quant_1         NxQxX1xY1       quantized color, half resolution
#   data_ab_pboost_1        Nx1xX1xY1       boost factor, half resolution
class ILSVRCTrnLabGaussianDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 0:
            raise Exception("Data layer should not have inputs")
        self.verbose = True
        self.X = 192
        self.Y = self.X
        # self.N = 84 # batch size
        # self.N = 10 # batch size
        # self.N = 64 # batch size
        self.N = 30
        self.B = 129 # total number of files
        # self.TO_PROD_SAME = True # keep spitting out the first minibatch
        self.TO_PROD_SAME = False
        self.NN = 10 # nearest neighbors
        self.SS = 2
        self.alpha = 0.25 # prior probability correction factor
        # self.alpha = 0.5 # prior probability correction factor
        self.sigma = 5 # sigma
        self.Lmax = 5 # maximum depth (exclusive)
        self.Lmin = 2 # minimum depth (inclusive), 0 for full-resolution

        self.Ldelt = self.Lmax-self.Lmin
        self.ildl = ILSVRCLabDataLoader(B=self.B,N=self.N,sameMode=self.TO_PROD_SAME, fileprefix='trn_data_lab_randord_224',verbose=self.verbose,filedir='/data/efros/rzhang/datasets/ILSVRC2012_processed')
        self.gp = cq.GaussianPyramid(Lmax=self.Lmax,Lmin=self.Lmin)
        self.nnenc = cq.NNEncode(self.NN,self.sigma,km_filepath='./data/color_bins/pts_in_hull.npy')
        self.pc = PriorFactor(self.alpha,priorFile='./data/color_bins/prior_probs.npy')

    def reshape(self, bottom, top):
        top[0].reshape(self.N,1,self.X,self.Y)
        for ll in range(self.Ldelt):
        # dd = 0
            top[1+ll*2].reshape(self.N,self.nnenc.K,self.X/self.SS**(ll+self.Lmin),self.Y/self.SS**(ll+self.Lmin))
            top[2+ll*2].reshape(self.N,1,self.X/self.SS**(ll+self.Lmin),self.Y/self.SS**(ll+self.Lmin))
 
    def forward(self, bottom, top):
        # self.alpha = 0.25
        # self.pc = PriorFactor(self.alpha,priorFile='/home/eecs/rich.zhang/src/projects/cross_domain/save/prior_prob_10/prior_probs.npy')
        data_lab = self.ildl.forward(self.X,self.Y,SS=1,returnAB=True,returnConcat=True)
        # py_data_lab = self.gp.pyramid_data_lab(data_lab)
        py_data_lab = self.gp.forward(data_lab)

        # print self.NN
        top[0].data[...] = data_lab[:,[0],:,:]
        for ll in range(self.Ldelt):
            top[1+ll*2].data[...] = self.nnenc.encode_points_mtx_nd(py_data_lab[ll][:,1:,:,:],axis=1)
            top[2+ll*2].data[...] = self.pc.forward(top[1+ll*2].data[...],axis=1)
        # dd = 0
        # top[1+dd*2].data[...] = self.nnenc.encode_points_mtx_nd(py_data_lab[:,1:,:,:],axis=1)
        # top[2+dd*2].data[...] = self.pc.forward(top[1+dd*2].data[...],axis=1)

    def backward(self, top, propagate_down, bottom):
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)

# New method of writing these data layers
# This is now a wrapper around data layer ILSVRCLabDataLoader and appropriate encoder
class ILSVRCTrnLabQuantNNDataLayer2(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 0:
            raise Exception("Data layer should not have inputs")

        # Data loader
        self.verbose = True
        # self.X = 224
        # self.N = 68 # batch size
        # self.X = 200
        # self.N = 84 # batch size
        self.X = 184
        # self.N = 100 # batch size
        self.N = 95
        self.Y = self.X
        self.B = 129 # total number of files
        self.verbose = True
        # self.TO_PROD_SAME = True # keep spitting out the first minibatch
        self.TO_PROD_SAME = False # keep spitting out the first minibatch
        self.NN = 10 # nearest neighbors
        self.sigma = 5 # sigma
        self.ildl = ILSVRCLabDataLoader(B=self.B,N=self.N,sameMode=self.TO_PROD_SAME, fileprefix='trn_data_lab_randord_224',verbose=self.verbose,filedir='/data/efros/rzhang/datasets/ILSVRC2012_processed')
        # self.ilpl = ILSVRCLabPaletteLoader(B=self.B,N=self.N,sameMode=self.TO_PROD_SAME, fileprefix='trn_data_randord_palette',verbose=self.verbose,filedir='/data/efros/rzhang/datasets/ILSVRC2012_processed')
        self.nnenc = cq.NNEncode(self.NN,self.sigma,km_filepath='./data/color_bins/pts_in_hull.npy')

    def reshape(self, bottom, top):
        top[0].reshape(self.N,1,self.X,self.Y)
        top[1].reshape(self.N,self.nnenc.K,self.X/self.SS,self.Y/self.SS)
 
    def forward(self, bottom, top):
        (data_l,data_ab) = self.ildl.forward(self.X,self.Y)

        # print self.NN
        top[0].data[...] = data_l
        top[1].data[...] = self.nnenc.encode_points_mtx_nd(data_ab,axis=1)
        # print t.tocStr()

    def backward(self, top, propagate_down, bottom):
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)

class ILSVRCTrnLabPaletteQuantLabelDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 0:
            raise Exception("Data layer should not have inputs")
        self.verbose = True
        # self.X = 184
        self.X = 224
        # self.N = 95
        self.N = 48
        self.Y = self.X
        self.B = 129 # total number of files
        self.verbose = True
        # keep spitting out the first minibatch
        self.TO_PROD_SAME = True
        # self.TO_PROD_SAME = False
        self.NN = 10 # nearest neighbors
        self.sigma = 5 # sigma
        self.K = 1000
        self.ildl = ILSVRCLabDataLoader(B=self.B,N=self.N,sameMode=self.TO_PROD_SAME,fileprefix='trn_data_lab_randord_224',filedir='/data/efros/rzhang/datasets/ILSVRC2012_processed',verbose=self.verbose)
        # self.ilpl = ILSVRCLabPaletteLoader(B=self.B,N=self.N,sameMode=self.TO_PROD_SAME, fileprefix='trn_data_randord_palette',verbose=self.verbose,filedir='/data/efros/rzhang/datasets/ILSVRC2012_processed')
        # self.nnenc = cq.NNEncode(self.NN,self.sigma,km_filepath='/home/eecs/rich.zhang/data_rzhang/models/caffe/cross_domain/l_to_ab/2015_02_13_classification_nn_rbf_reggrid/pts_in_hull.npy')
        self.ilpll = ILSVRCLabPaletteLabelLoader(K=self.K,N=self.N,sameMode=self.TO_PROD_SAME, verbose=self.verbose)

    def reshape(self, bottom, top):
        top[0].reshape(self.N,1,self.X,self.Y)
        # top[1].reshape(self.N,self.nnenc.K,self.X/self.SS,self.Y/self.SS)
        top[1].reshape(self.N,1,1,1)
 
    def forward(self, bottom, top):
        t = rz.Timer()

        cur_db = self.ildl.cur_db
        cur_cnt = self.ildl.cur_cnt

        data_l = self.ildl.forward(self.X,self.Y,returnAB=False)
        top[0].data[...] = data_l
        top[1].data[:,0,0,0] = self.ilpll.forward(override_db=cur_db,override_cnt=cur_cnt)
        # print data_l.flatten()[0:10]
        # print top[1].data

    def backward(self, top, propagate_down, bottom):
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)

class ILSVRCLabPaletteLoader():
    def __init__(self,B=129,N=68,sameMode=False,verbose=True,filedir='/data/efros/rzhang/datasets/ILSVRC2012_processed',fileprefix='trn_data_randord_palette'):
        self.hdf5filedir = filedir
        self.fileprefix = fileprefix
        self.verbose = verbose
        self.TO_PROD_SAME = sameMode # keep spitting out the first minibatch

        if(self.verbose):
            print('Loading ILSVRC training palette data')
        # self.N = 96 # batch size
        self.N = N # batch size
        self.B = B # total number of files

        self.t = rz.Timer()
        if(self.verbose):
            print('  Subdatabases found: %i'%self.B)
        self.hdf5filepaths = np.zeros(self.B,dtype='object')
        self.hdf5s = np.zeros(self.B,dtype='object')
        self.data = np.zeros(self.B,dtype='object')
        self.Ns = np.zeros(self.B,dtype='int')
        for bb in range(self.B):
            self.hdf5filepaths[bb] = os.path.join(self.hdf5filedir, '%s_%i.h5'%(fileprefix,bb))
            self.hdf5s[bb] = rz.load_from_hdf5(self.hdf5filepaths[bb])
            self.data[bb] = self.hdf5s[bb]['cc_ab']
            self.Ns[bb] = self.data[bb].shape[0]
        self.Ntotal = np.sum(self.Ns)
        self.C = self.data[0].shape[2]
        self.cur_db = 0
        self.cur_cnt = 0
 
    def forward(self,override_db=-1,override_cnt=-1): # spit out data
        if(not rz.check_value(override_db,-1)):
            self.cur_db = override_db
        if(not rz.check_value(override_cnt,-1)):
            self.cur_cnt = override_cnt

        t = rz.Timer()

        db_inds = {}
        cnt_inds = {}
        if(self.cur_cnt+self.N >= self.Ns[self.cur_db]):
            db_inds[0] = self.cur_db
            cnt_inds[0] = np.arange(self.cur_cnt,self.Ns[self.cur_db])
            self.cur_cnt = self.N-(self.Ns[self.cur_db]-self.cur_cnt)
            self.cur_db = np.mod(self.cur_db+1,self.B)
            if(self.cur_cnt!=0):
                db_inds[1] = self.cur_db
                cnt_inds[1] = np.arange(0,self.cur_cnt)
        else:
            db_inds[0] = self.cur_db
            cnt_inds[0] = self.cur_cnt+np.arange(0,self.N)
            if(not self.TO_PROD_SAME): # assuming first file is bigger than a minibatch
                self.cur_cnt = self.cur_cnt+self.N # accumulate count
            else:
                self.cur_cnt = 0 # do not accumulate, just produce same thing

        if(self.verbose):
            for cc in range(len(cnt_inds)):
                print('%s: %i: inds %i-%i from db %i'%(self.t.tocStr(),cc,cnt_inds[cc][0],cnt_inds[cc][-1],db_inds[cc]))

        # fill in data from the databases
        data_palette = np.zeros((self.N,4,self.C),dtype='float32')

        cnt = 0
        for cc in range(len(cnt_inds)):
            inds = cnt+np.arange(0,cnt_inds[cc].size) # indices into data layer
            cnt = cnt+cnt_inds[cc].size
            db = db_inds[cc]
            data_palette[inds,:,:] = self.data[db][cnt_inds[cc],:,:]
        return data_palette

class ILSVRCLabPaletteLabelLoader():
    def __init__(self,K=100,N=95,sameMode=False,verbose=False,fileprefix='trn_palette_k.npy',filedir='/home/eecs/rich.zhang/src/projects/cross_domain/save/palette'):
        # self.ilpl = ILSVRCLabPaletteLoader()
        self.N = N
        self.verbose = True
        self.S = 1e4 # equivalent chunking for the LAB images
        self.K = K
        self.filedir = '%s_%i'%(filedir,K)
        self.fileprefix = fileprefix
        self.trn_palette_k = np.load(os.path.join(self.filedir,self.fileprefix))
        self.cur_db = 0
        self.cur_cnt = 0
        self.cur_cnt_tot = 0
        self.N_tot = self.trn_palette_k.size
        self.t = rz.Timer()
        self.TO_PROD_SAME = sameMode

    def forward(self,override_db=-1,override_cnt=-1):
        if(not rz.check_value(override_db,-1)):
            self.cur_db = override_db
        if(not rz.check_value(override_cnt,-1)):
            self.cur_cnt = override_cnt

        if(self.verbose):
            print('%s: db-%i, cnt-%i'%(self.t.tocStr(),self.cur_db,self.cur_cnt))

        self.cur_cnt = self.S*override_db + override_cnt
        # print self.trn_palette_k
        ret_val = self.trn_palette_k[np.mod(np.arange(self.cur_cnt_tot,self.cur_cnt_tot+self.N),self.N_tot)]

        if(not self.TO_PROD_SAME):
            self.cur_cnt_tot+=self.N
            self.cur_cnt_tot = np.mod(self.cur_cnt_tot,self.N_tot)

        self.cur_db = self.cur_cnt_tot/self.S
        self.cur_cnt = np.mod(self.cur_cnt_tot,self.S)

        return ret_val

class ILSVRCLabDataLoader():
    def __init__(self,B=129,N=68,sameMode=False,verbose=True,filedir='/data/efros/rzhang/datasets/ILSVRC2012_processed',fileprefix='trn_data_lab_randord_224'):
        self.hdf5filedir = filedir
        self.fileprefix = fileprefix
        self.verbose = verbose
        self.TO_PROD_SAME = sameMode # keep spitting out the first minibatch

        if(self.verbose):
            print('Loading ILSVRC training LAB data')
        # self.N = 96 # batch size
        self.N = N # batch size
        self.B = B # total number of files

        self.t = rz.Timer()
        if(self.verbose):
            print('  Subdatabases found: %i'%self.B)
        self.hdf5filepaths = np.zeros(self.B,dtype='object')
        self.hdf5s = np.zeros(self.B,dtype='object')
        self.data = np.zeros(self.B,dtype='object')
        self.Ns = np.zeros(self.B,dtype='int')
        for bb in range(self.B):
            self.hdf5filepaths[bb] = os.path.join(self.hdf5filedir, '%s_%i.h5'%(fileprefix,bb))
            self.hdf5s[bb] = rz.load_from_hdf5(self.hdf5filepaths[bb])
            self.data[bb] = self.hdf5s[bb]['data']
            self.Ns[bb] = self.data[bb].shape[0]
        self.Ntotal = np.sum(self.Ns)
        self.X = self.data[0].shape[2]
        self.Y = self.data[0].shape[3]
        self.cur_db = np.random.randint(self.B)
        # self.cur_db = 51
        self.cur_cnt = 0
 
    def forward(self,X=224,Y=224,SS=8,randJitter=True,randFlip=True,override_db=-1,override_cnt=-1,returnAB=True,returnConcat=False,returnGrayMask=False): # spit out data
        # INPUTS
        #   X,Y
        #   SS
        #   randJitter
        #   randFlip
        #   override_db
        #   override_cnt
        #   returnAB
        #   returnConcat
        #   returnGrayMask
        # OUTPUTS
        #   data_l
        #   data_ab
        #   grayMask

        # self.TO_PROD_SAME = False

        if(not rz.check_value(override_db,-1)):
            self.cur_db = override_db
        if(not rz.check_value(override_cnt,-1)):
            self.cur_cnt = override_cnt

        t = rz.Timer()

        db_inds = {}
        cnt_inds = {}
        if(self.cur_cnt+self.N >= self.Ns[self.cur_db]):
            db_inds[0] = self.cur_db
            cnt_inds[0] = np.arange(self.cur_cnt,self.Ns[self.cur_db])
            self.cur_cnt = self.N-(self.Ns[self.cur_db]-self.cur_cnt)
            self.cur_db = np.mod(self.cur_db+1,self.B)
            if(self.cur_cnt!=0):
                db_inds[1] = self.cur_db
                cnt_inds[1] = np.arange(0,self.cur_cnt)
        else:
            db_inds[0] = self.cur_db
            cnt_inds[0] = self.cur_cnt+np.arange(0,self.N)
            if(not self.TO_PROD_SAME): # assuming first file is bigger than a minibatch
                self.cur_cnt = self.cur_cnt+self.N # accumulate count
            else:
                self.cur_cnt = 0 # do not accumulate, just produce same thing

        if(self.verbose):
            for cc in range(len(cnt_inds)):
                print('%s: %i: inds %i-%i from db %i'%(self.t.tocStr(),cc,cnt_inds[cc][0],cnt_inds[cc][-1],db_inds[cc]))

        # fill in data from the databases
        data_l = np.zeros((self.N,1,X,Y),dtype='float32')
        if(returnAB):
            data_ab = np.zeros((self.N,2,X/SS,Y/SS),dtype='float32')

        # random jittering
        if(randJitter):
            randoffX = np.random.randint(0,self.X-X+1)
            randoffY = np.random.randint(0,self.Y-Y+1)
        else:
            randoffX = 0
            randoffY = 0

        cnt = 0
        for cc in range(len(cnt_inds)):
            inds = cnt+np.arange(0,cnt_inds[cc].size) # indices into data layer
            cnt = cnt+cnt_inds[cc].size
            db = db_inds[cc]

            if(cnt_inds[cc].flatten().size==1):
                data_l[[inds],:,:,:] = self.data[db][cnt_inds[cc],:,:,:][rz.na(),[0],randoffX:randoffX+X,randoffY:randoffY+X]
                if(returnAB):
                    data_ab[[inds],:,:,:] = self.data[db][cnt_inds[cc],:,:,:][rz.na(),1:,randoffX:randoffX+X:SS,randoffY:randoffY+Y:SS]
            else:
                data_lab = self.data[db][cnt_inds[cc],:,:,:]
                data_l[inds,:,:,:] = data_lab[:,[0],randoffX:randoffX+X,randoffY:randoffY+X]
                if(returnAB):
                    data_ab[inds,:,:,:] = data_lab[:,1:,randoffX:randoffX+X:SS,randoffY:randoffY+Y:SS]

        # random flipping
        if(randFlip):
            if(np.random.rand()>.5): # flip
                data_l[...] = data_l[:,:,::-1,:]
                if(returnAB):
                    data_ab[...] = data_ab[:,:,::-1,:]

        if(returnGrayMask):
            gray_mask = cq.is_data_lab_gray(data_ab)

        if(returnAB):
            if(returnConcat):
                if(returnGrayMask):
                    return (np.concatenate((data_l,data_ab),axis=1),gray_mask)
                else:
                    return np.concatenate((data_l,data_ab),axis=1)
            else:
                if(returnGrayMask):
                    return (data_l,data_ab,gray_mask)
                else:
                    return (data_l,data_ab)
        else:
            if(returnGrayMask):
                return (data_l,gray_mask)
            else:
                return data_l

def flatten_nxy(in_array):
    (N,M,X,Y) = in_array.shape
    return in_array.transpose((0,2,3,1)).reshape((N*X*Y,M))

def unflatten_nxy(in_array,N,X,Y):
    M = in_array.shape[1]
    return in_array.reshape((N,X,Y,M)).transpose(N,M,X,Y)

# Load training data in LAB space
# Should be in '/data/efros/rzhang/datasets/ILSVRC2012_processed'
#   /trn_data_lab_randord_224_%i.h5
#   /trn_data_ab_quant_20_ss_4_randord_224_0.h5
#       quantized with grid size of 20
#       quantification subsampled by factor of 4
#   binned together in 129 files with 10k files each
#   load self.N per batch
# class ILSVRCTrnLabQuant20SS4DataLayer(caffe.Layer):
class ILSVRCTrnLabQuantInd20SS4DataLayer(caffe.Layer):
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 0:
            raise Exception("Data layer should not have inputs")

        self.verbose = True
        if(self.verbose):
            print('Loading ILSVRC training LAB data')
        self.N = 60 # batch size
        self.B = 129 # total number of files
        # self.N = 64 # batch size
        # self.N = 2
        # self.B = 1 # total number of files
        self.SSI = 4 # subsample factor on input
        self.SSO = 2 # subsample on output
        self.SS = self.SSI*self.SSO
        self.GRID = 20
        self.verbose = True
        self.t = rz.Timer()
        self.TO_PROD_SAME = True # keep spitting out the first minibatch
        # self.TO_PROD_SAME = False # keep spitting out the first minibatch
        self.PRODUCE_TOP_IND = True # only produce top index
        # self.PRODUCE_TOP_IND = False # only produce top index

        self.load_overhead()

    def load_overhead(self):
        self.hdf5filedir = '/data/efros/rzhang/datasets/ILSVRC2012_processed'
        if(self.verbose):
            print('  Subdatabases found: %i'%self.B)
        self.hdf5filepaths = np.zeros(self.B,dtype='object')
        self.hdf5s = np.zeros(self.B,dtype='object')
        self.data = np.zeros(self.B,dtype='object')
        self.quant_hdf5filepaths = np.zeros(self.B,dtype='object')
        self.quant_hdf5s = np.zeros(self.B,dtype='object')
        self.quant_inds = np.zeros(self.B,dtype='object')
        self.quant_wts = np.zeros(self.B,dtype='object')
        self.Ns = np.zeros(self.B,dtype='int')

        for bb in range(self.B):
            self.hdf5filepaths[bb] = os.path.join(self.hdf5filedir, 'trn_data_lab_randord_224_%i.h5'%(bb))
            self.hdf5s[bb] = rz.load_from_hdf5(self.hdf5filepaths[bb])
            self.data[bb] = self.hdf5s[bb]['data']
            self.Ns[bb] = self.data[bb].shape[0]
            # self.Ns[bb] = 127

            self.quant_hdf5filepaths[bb] = os.path.join(self.hdf5filedir, 'trn_data_ab_quant_%i_ss_%i_randord_224_%i.h5'%(self.GRID,self.SSI,bb))
            self.quant_hdf5s[bb] = rz.load_from_hdf5(self.quant_hdf5filepaths[bb])
            self.quant_inds[bb] = self.quant_hdf5s[bb]['inds']
            self.quant_wts[bb] = self.quant_hdf5s[bb]['wts']
            self.AB = self.quant_hdf5s[bb]['AB'][0]

        self.Ntotal = np.sum(self.Ns)
        self.X = self.data[0].shape[2]
        self.Y = self.data[0].shape[3]
        self.XSSI = self.quant_inds[0].shape[2] # input resolution of quantized ab
        self.YSSI = self.quant_inds[0].shape[3]
        self.XSSO = self.X/self.SS # output resolution of quantized ab loss
        self.YSSO = self.Y/self.SS
        self.cur_db = 0
        self.cur_cnt = 0

    def reshape(self, bottom, top):
        top[0].reshape(self.N,1,self.X,self.Y)

        if(self.PRODUCE_TOP_IND):
            top[1].reshape(self.N,1,self.X/self.SS,self.Y/self.SS) # currently hard-coded
        else:
            top[1].reshape(self.N,144,self.X/self.SS,self.Y/self.SS) # currently hard-coded
 
    def forward(self, bottom, top):
        # self.cur_db = 1
        # self.cur_cnt = 0

        # figure out which indices to sample from
        db_inds = {}
        cnt_inds = {}
        if(self.cur_cnt+self.N >= self.Ns[self.cur_db]):
            db_inds[0] = self.cur_db
            cnt_inds[0] = np.arange(self.cur_cnt,self.Ns[self.cur_db])

            # accumulate
            self.cur_cnt = self.N-(self.Ns[self.cur_db]-self.cur_cnt)

            self.cur_db = np.mod(self.cur_db+1,self.B)
            if(self.cur_cnt!=0):
                db_inds[1] = self.cur_db
                cnt_inds[1] = np.arange(0,self.cur_cnt)
        else:
            db_inds[0] = self.cur_db
            cnt_inds[0] = self.cur_cnt+np.arange(0,self.N)

            if(not self.TO_PROD_SAME): # assuming first file is bigger than a minibatch
                self.cur_cnt = self.cur_cnt+self.N # accumulate count
            else:
                self.cur_cnt = 0

        if(self.verbose):
        # if(1):
            for cc in range(len(cnt_inds)):
                print('%s: %i: inds %i-%i from db %i'%(self.t.tocStr(),cc,cnt_inds[cc][0],cnt_inds[cc][-1],db_inds[cc]))

        # # fill in data from the databases
        cnt = 0
        for cc in range(len(cnt_inds)):
            inds = cnt+np.arange(0,cnt_inds[cc].size) # indices into data layer
            cnt = cnt+cnt_inds[cc].size
            db = db_inds[cc]
            cur_size = cnt_inds[cc].flatten().size

            # print cnt_inds[cc].flatten().size
            if(cur_size==1):
                top[0].data[inds,:,:,:] = self.data[db][cnt_inds[cc],:,:,:][rz.na(),[0],:,:]
                quant_inds = self.quant_inds[db][[cnt_inds[cc]],:,::self.SSO,::self.SSO]
                quant_wts = self.quant_wts[db][[cnt_inds[cc]],:,::self.SSO,::self.SSO]
                # top[1].data[inds,:,:,:] = self.data[db][cnt_inds[cc],:,:,:][rz.na(),1:,::self.SS,::self.SS]
            else:
                top[0].data[inds,:,:,:] = self.data[db][cnt_inds[cc].flatten(),:,:,:][:,[0],:,:]
                quant_inds = self.quant_inds[db][cnt_inds[cc],:,::self.SSO,::self.SSO]
                quant_wts = self.quant_wts[db][cnt_inds[cc],:,::self.SSO,::self.SSO]
                # top[1].data[inds,:,:,:] = self.data[db][cnt_inds[cc].flatten(),:,:,:][:,1:,::self.SS,::self.SS]

            if(self.PRODUCE_TOP_IND): # just find top index
                (NC,CC,XC,YC) = quant_wts.shape
                max_inds = np.argmax(quant_wts,axis=1)
                out_inds = np.zeros((NC,1,XC,YC))

                t = rz.Timer()
                for nn in range(NC):
                    for xx in range(XC):
                        for yy in range(YC):
                            # print max_inds[nn,xx,yy]
                            # print max_inds
                            out_inds[nn,0,xx,yy] = quant_inds[nn,max_inds[nn,xx,yy],xx,yy]
                # print t.tocStr()

                top[1].data[inds,:,:,:] = out_inds
            else:
                quant_imginds = np.arange(4*cur_size*self.XSSO*self.YSSO)/4
                quant_inds = quant_inds.transpose((0,2,3,1)).reshape((self.XSSO*self.YSSO*cur_size,4))
                quant_wts = quant_wts.transpose((0,2,3,1)).reshape((self.XSSO*self.YSSO*cur_size,4))
                quant_mtx = np.zeros((cur_size*self.XSSO*self.YSSO,self.AB),dtype='float32')
                quant_mtx[quant_imginds.flatten(), quant_inds.flatten()] = quant_wts.flatten()
                quant_mtx = quant_mtx.reshape((cur_size,self.XSSO,self.YSSO,self.AB)).transpose((0,3,1,2))
                top[1].data[inds,:,:,:] = quant_mtx

    def backward(self, top, propagate_down, bottom):
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)

class MaskImage32DataLayer(caffe.Layer):
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 0:
            raise Exception("Data layer should not have inputs")

        self.N = 256 # batch size
        # print 'Batchsize = %i'%self.N
        self.load_overhead()

    def load_overhead(self):
        self.hdf5filedir = '/home/eecs/rich.zhang/data_rzhang/datasets/cross_domain/mask_to_image_32'
        self.hdf5filepath = os.path.join(self.hdf5filedir, 'maskImage32_vehicle.h5')
        self.hdf5 = rz.load_from_hdf5(self.hdf5filepath)
        self.mask = self.hdf5['mask']
        self.Ntotal = self.mask.shape[0]
        self.X = self.mask.shape[2]
        self.Y = self.mask.shape[3]
        self.cnt = 0

    def reshape(self, bottom, top):
        top[0].reshape(self.N,1,self.X,self.Y)
        top[1].reshape(self.N,3,self.X,self.Y)
 
    def forward(self, bottom, top):
        if(self.cnt+self.N >= self.Ntotal):
            # self.Ntotal-self.cnt
            top[0].data[:self.Ntotal-self.cnt,:,:,:] = self.hdf5['mask'][self.cnt:,:,:,:]
            top[0].data[self.Ntotal-self.cnt:,:,:,:] = self.hdf5['mask'][:self.N-(self.Ntotal-self.cnt),:,:,:]
            top[1].data[:self.Ntotal-self.cnt,:,:,:] = self.hdf5['img_mask'][self.cnt:,:,:,:]
            top[1].data[self.Ntotal-self.cnt:,:,:,:] = self.hdf5['img_mask'][:self.N-(self.Ntotal-self.cnt),:,:,:]
        else:
            top[0].data[...] = self.hdf5['mask'][self.cnt:self.cnt+self.N,:,:,:]
            top[1].data[...] = self.hdf5['img_mask'][self.cnt:self.cnt+self.N,:,:,:]
        self.cnt = np.mod(self.cnt+self.N, self.Ntotal)
        # print 'Current count: %i'%self.cnt
        # print 'Current count, N: %i, %i'%(self.cnt,self.Ntotal)

    def backward(self, top, propagate_down, bottom):
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)

# Load training data in LAB space
# Should be in '/data/efros/rzhang/datasets/ILSVRC2012_processed'
#   /trn_data_lab_randord_224_%i.h5
#   binned together in 129 files with 10k files each
#   load self.N per batch
# class ILSVRCTrnLabDataLayer(caffe.Layer):
#     def setup(self, bottom, top):
#         # check input pair
#         if len(bottom) != 0:
#             raise Exception("Data layer should not have inputs")

#         self.verbose = True
#         if(self.verbose):
#             print 'Loading ILSVRC training LAB data'
#         # self.N = 96 # batch size
#         self.N = 64 # batch size
#         # self.B = 1 # total number of files
#         self.B = 129 # total number of files
#         self.verbose = True
#         self.t = rz.Timer()
#         self.TO_NORM_AB = True # divide output by 110
#         self.TO_PROD_SAME = False # keep spitting out the first minibatch
#         self.load_overhead()

#     def load_overhead(self):
#         self.hdf5filedir = '/data/efros/rzhang/datasets/ILSVRC2012_processed'
#         if(self.verbose):
#             print '  Subdatabases found: %i'%self.B
#         self.hdf5filepaths = np.zeros(self.B,dtype='object')
#         self.hdf5s = np.zeros(self.B,dtype='object')
#         self.data = np.zeros(self.B,dtype='object')
#         self.Ns = np.zeros(self.B,dtype='int')
#         for bb in range(self.B):
#             self.hdf5filepaths[bb] = os.path.join(self.hdf5filedir, 'trn_data_lab_randord_224_%i.h5'%(bb))
#             self.hdf5s[bb] = rz.load_from_hdf5(self.hdf5filepaths[bb])
#             self.data[bb] = self.hdf5s[bb]['data']
#             self.Ns[bb] = self.data[bb].shape[0]
#         self.Ntotal = np.sum(self.Ns)
#         self.X = self.data[0].shape[2]
#         self.Y = self.data[0].shape[3]
#         self.cur_db = 0
#         self.cur_cnt = 0
#         self.SS = 8 # subsample on output

#     def reshape(self, bottom, top):
#         top[0].reshape(self.N,1,self.X,self.Y)
#         top[1].reshape(self.N,2,self.X/self.SS,self.Y/self.SS)
 
#     def forward(self, bottom, top):
#         # self.cur_db = 0
#         # self.cur_cnt = 9952
#         # print 'hello'

#         # figure out which indices to sample from
#         db_inds = {}
#         cnt_inds = {}
#         if(self.cur_cnt+self.N >= self.Ns[self.cur_db]):
#             db_inds[0] = self.cur_db
#             cnt_inds[0] = np.arange(self.cur_cnt,self.Ns[self.cur_db])

#             # accumulate
#             # print self.N
#             # print self.Ns[self.cur_db]
#             # print self.cur_cnt
#             self.cur_cnt = self.N-(self.Ns[self.cur_db]-self.cur_cnt)
#             # print self.cur_cnt

#             self.cur_db = np.mod(self.cur_db+1,self.B)
#             if(self.cur_cnt!=0):
#                 db_inds[1] = self.cur_db
#                 cnt_inds[1] = np.arange(0,self.cur_cnt)
#         else:
#             db_inds[0] = self.cur_db
#             cnt_inds[0] = self.cur_cnt+np.arange(0,self.N)

#             if(not self.TO_PROD_SAME): # assuming first file is bigger than a minibatch
#                 self.cur_cnt = self.cur_cnt+self.N # accumulate count
#             else:
#                 self.cur_cnt = 0

#         if(self.verbose):
#         # if(1):
#             for cc in range(len(cnt_inds)):
#                 print '%s: %i: inds %i-%i from db %i'%(self.t.tocStr(),cc,cnt_inds[cc][0],cnt_inds[cc][-1],db_inds[cc])
#                 # cc
#                 # cnt_inds[cc][0]
#                 # cnt_inds[cc][1]
#                 # db_inds[cc]

#         # # fill in data from the databases
#         cnt = 0
#         for cc in range(len(cnt_inds)):
#             inds = cnt+np.arange(0,cnt_inds[cc].size) # indices into data layer
#             cnt = cnt+cnt_inds[cc].size
#             db = db_inds[cc]
            
#             # print cnt_inds[cc].flatten().size
#             if(cnt_inds[cc].flatten().size==1):
#                 # print 'hello'
#                 # print top[0].data[inds,:,:,:].shape
#                 # print cnt_inds[cc]
#                 # print self.data[db][cnt_inds[cc],:,:,:].shape
#                 top[0].data[inds,:,:,:] = self.data[db][cnt_inds[cc],:,:,:][rz.na(),[0],:,:]
#                 top[1].data[inds,:,:,:] = self.data[db][[cnt_inds[cc]],:,:,:][rz.na(),1:,::self.SS,::self.SS]
#             else:
#                 top[0].data[inds,:,:,:] = self.data[db][cnt_inds[cc].flatten(),:,:,:][:,[0],:,:]
#                 top[1].data[inds,:,:,:] = self.data[db][cnt_inds[cc].flatten(),:,:,:][:,1:,::self.SS,::self.SS]
#             # print top[0].data[inds,:,:,:].shape
#             # print self.data[db][cnt_inds[cc],:,:,:].shape

#         if(self.TO_NORM_AB):
#             top[1].data[...] = top[1].data[...]/110.

#         # print self.Ns[self.cur_db]
#         # print self.cur_db
#         # print self.cur_cnt

    def backward(self, top, propagate_down, bottom):
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)

class ILSVRCValLabDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 0:
            raise Exception("Data layer should not have inputs")

        self.N = 256 # batch size
        self.verbose = True
        self.SS = 8
        self.load_overhead()

    def load_overhead(self):
        self.hdf5filedir = '/data/efros/rzhang/datasets/ILSVRC2012_processed'
        # self.hdf5filepath = os.path.join(self.hdf5filedir, 'val_data_lab_randord_224.h5')
        self.hdf5filepath = os.path.join(self.hdf5filedir, 'val_data_lab_randord_224.h5')
        self.hdf5 = rz.load_from_hdf5(self.hdf5filepath)
        self.data = self.hdf5['data']
        self.Ntotal = self.data.shape[0]
        self.X = self.data.shape[2]
        self.Y = self.data.shape[3]
        self.cnt = 0

    def reshape(self, bottom, top):
        top[0].reshape(self.N,1,self.X,self.Y)
        top[1].reshape(self.N,2,self.X/self.SS,self.Y/self.SS)
        # top[0].reshape(self.N,3,self.X,self.Y)
 
    def forward(self, bottom, top):
        if(self.cnt+self.N >= self.Ntotal):
            # self.Ntotal-self.cnt
            top[0].data[:self.Ntotal-self.cnt,:,:,:] = self.hdf5['data'][self.cnt:,[0],:,:]
            top[0].data[self.Ntotal-self.cnt:,:,:,:] = self.hdf5['data'][:self.N-(self.Ntotal-self.cnt),[0],:,:]
            top[1].data[:self.Ntotal-self.cnt,:,:,:] = self.hdf5['data'][self.cnt:,1:,::self.SS,::self.SS]
            top[1].data[self.Ntotal-self.cnt:,:,:,:] = self.hdf5['data'][:self.N-(self.Ntotal-self.cnt),1:,::self.SS,::self.SS]
        else:
            top[0].data[...] = self.hdf5['data'][self.cnt:self.cnt+self.N,[0],:,:]
            top[1].data[...] = self.hdf5['data'][self.cnt:self.cnt+self.N,1:,::self.SS,::self.SS]
            # top[0].data[...] = self.hdf5['data'][self.cnt:self.cnt+self.N,:,:,:]
        self.cnt = np.mod(self.cnt+self.N, self.Ntotal)
        # print 'Current count: %i'%self.cnt
        # print 'Current count, N: %i, %i'%(self.cnt,self.Ntotal)

    def backward(self, top, propagate_down, bottom):
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)

# Load training data in LAB space
# Should be in '/data/efros/rzhang/datasets/ILSVRC2012_processed'
#   /trn_data_lab_randord_224_%i.h5
#   binned together in 129 files with 10k files each
#   load self.N per batch
class ILSVRCTrnLabQuantIndDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 0:
            raise Exception("Data layer should not have inputs")

        self.verbose = True
        if(self.verbose):
            print('Loading ILSVRC training LAB data')
        # self.N = 96 # batch size
        self.N = 60 # batch size
        # self.B = 1 # total number of files
        self.B = 129 # total number of files
        self.verbose = True
        self.t = rz.Timer()
        # self.TO_NORM_AB = True # divide output by 110
        # self.TO_PROD_SAME = True # keep spitting out the first minibatch
        self.TO_PROD_SAME = False # keep spitting out the first minibatch
        self.load_overhead()
        self.SS = 8 # subsample on output

    def load_overhead(self):
        self.hdf5filedir = '/data/efros/rzhang/datasets/ILSVRC2012_processed'
        if(self.verbose):
            print('  Subdatabases found: %i'%self.B)
        self.hdf5filepaths = np.zeros(self.B,dtype='object')
        self.hdf5s = np.zeros(self.B,dtype='object')
        self.data = np.zeros(self.B,dtype='object')
        self.Ns = np.zeros(self.B,dtype='int')
        for bb in range(self.B):
            self.hdf5filepaths[bb] = os.path.join(self.hdf5filedir, 'trn_data_lab_randord_224_%i.h5'%(bb))
            self.hdf5s[bb] = rz.load_from_hdf5(self.hdf5filepaths[bb])
            self.data[bb] = self.hdf5s[bb]['data']
            self.Ns[bb] = self.data[bb].shape[0]
        self.Ntotal = np.sum(self.Ns)
        self.X = self.data[0].shape[2]
        self.Y = self.data[0].shape[3]
        self.cur_db = 0
        self.cur_cnt = 0
        self.grid_49 = rz.grid_ab(40)
        self.grid_25 = rz.grid_ab(60)
        self.grid_16 = rz.grid_ab(80)
        self.grid_9 = rz.grid_ab(80)

    def reshape(self, bottom, top):
        top[0].reshape(self.N,1,self.X,self.Y)
        top[1].reshape(self.N,1,self.X/self.SS,self.Y/self.SS)
        top[2].reshape(self.N,1,self.X/self.SS,self.Y/self.SS)
        top[3].reshape(self.N,1,self.X/self.SS,self.Y/self.SS)
        top[4].reshape(self.N,1,self.X/self.SS,self.Y/self.SS)
 
    def forward(self, bottom, top):
        # self.cur_db = 0
        # self.cur_cnt = 9952
        # print 'hello'

        # figure out which indices to sample from
        db_inds = {}
        cnt_inds = {}
        if(self.cur_cnt+self.N >= self.Ns[self.cur_db]):
            db_inds[0] = self.cur_db
            cnt_inds[0] = np.arange(self.cur_cnt,self.Ns[self.cur_db])

            # accumulate
            # print self.N
            # print self.Ns[self.cur_db]
            # print self.cur_cnt
            self.cur_cnt = self.N-(self.Ns[self.cur_db]-self.cur_cnt)

            self.cur_db = np.mod(self.cur_db+1,self.B)
            if(self.cur_cnt!=0):
                db_inds[1] = self.cur_db
                cnt_inds[1] = np.arange(0,self.cur_cnt)
        else:
            db_inds[0] = self.cur_db
            cnt_inds[0] = self.cur_cnt+np.arange(0,self.N)

            if(not self.TO_PROD_SAME): # assuming first file is bigger than a minibatch
                self.cur_cnt = self.cur_cnt+self.N # accumulate count
            else:
                self.cur_cnt = 0

        if(self.verbose):
            for cc in range(len(cnt_inds)):
                print('%s: %i: inds %i-%i from db %i'%(self.t.tocStr(),cc,cnt_inds[cc][0],cnt_inds[cc][-1],db_inds[cc]))

        # fill in data from the databases
        data_ab = np.zeros((self.N,2,self.X/self.SS,self.Y/self.SS),dtype='float32')

        cnt = 0
        for cc in range(len(cnt_inds)):
            inds = cnt+np.arange(0,cnt_inds[cc].size) # indices into data layer
            cnt = cnt+cnt_inds[cc].size
            db = db_inds[cc]

            # print cnt_inds[cc].flatten().size
            if(cnt_inds[cc].flatten().size==1):
                top[0].data[inds,:,:,:] = self.data[db][cnt_inds[cc],:,:,:][rz.na(),[0],:,:]
                # top[1].data[inds,:,:,:] = self.data[db][[cnt_inds[cc]],:,:,:][rz.na(),1:,::self.SS,::self.SS]
                data_ab[inds,:,:,:] = self.data[db][[cnt_inds[cc]],:,:,:][rz.na(),1:,::self.SS,::self.SS]
            else:
                top[0].data[inds,:,:,:] = self.data[db][cnt_inds[cc].flatten(),:,:,:][:,[0],:,:]
                data_ab[inds,:,:,:] = self.data[db][cnt_inds[cc].flatten(),:,:,:][:,1:,::self.SS,::self.SS]
                # top[1].data[inds,:,:,:] = self.data[db][cnt_inds[cc].flatten(),:,:,:][:,1:,::self.SS,::self.SS]

        data_ab = data_ab.transpose((0,2,3,1)).reshape((self.N*self.X/self.SS*self.Y/self.SS,2))
        inds_enc = self.grid_49.encode_nearest_points(data_ab)
        inds_enc = inds_enc.reshape((self.N,1,self.X/self.SS,self.Y/self.SS))
        top[1].data[...] = inds_enc

        inds_enc = self.grid_25.encode_nearest_points(data_ab)
        inds_enc = inds_enc.reshape((self.N,1,self.X/self.SS,self.Y/self.SS))
        top[2].data[...] = inds_enc

        inds_enc = self.grid_16.encode_nearest_points(data_ab)
        inds_enc = inds_enc.reshape((self.N,1,self.X/self.SS,self.Y/self.SS))
        top[3].data[...] = inds_enc

        inds_enc = self.grid_9.encode_nearest_points(data_ab)
        inds_enc = inds_enc.reshape((self.N,1,self.X/self.SS,self.Y/self.SS))
        top[4].data[...] = inds_enc

        # if(self.TO_NORM_AB):
            # top[1].data[...] = top[1].data[...]/110.
        # if(self.TO_NORM_AB):

        # top[1].data[...] = top[1].data[...]/110.

    def backward(self, top, propagate_down, bottom):
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)

# Load training data in LAB space
# Should be in '/data/efros/rzhang/datasets/ILSVRC2012_processed'
#   /trn_data_lab_randord_224_%i.h5
#   binned together in 129 files with 10k files each
#   load self.N per batch
class ILSVRCTrnLabQuantMapDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 0:
            raise Exception("Data layer should not have inputs")

        self.verbose = True
        if(self.verbose):
            print('Loading ILSVRC training LAB data')
        # self.N = 96 # batch size
        self.N = 60 # batch size
        # self.B = 1 # total number of files
        self.B = 129 # total number of files
        self.verbose = True
        self.t = rz.Timer()
        # self.TO_NORM_AB = True # divide output by 110
        # self.TO_PROD_SAME = True # keep spitting out the first minibatch
        self.TO_PROD_SAME = False # keep spitting out the first minibatch
        self.load_overhead()
        self.SS = 8 # subsample on output

    def load_overhead(self):
        self.hdf5filedir = '/data/efros/rzhang/datasets/ILSVRC2012_processed'
        if(self.verbose):
            print('  Subdatabases found: %i'%self.B)
        self.hdf5filepaths = np.zeros(self.B,dtype='object')
        self.hdf5s = np.zeros(self.B,dtype='object')
        self.data = np.zeros(self.B,dtype='object')
        self.Ns = np.zeros(self.B,dtype='int')
        for bb in range(self.B):
            self.hdf5filepaths[bb] = os.path.join(self.hdf5filedir, 'trn_data_lab_randord_224_%i.h5'%(bb))
            self.hdf5s[bb] = rz.load_from_hdf5(self.hdf5filepaths[bb])
            self.data[bb] = self.hdf5s[bb]['data']
            self.Ns[bb] = self.data[bb].shape[0]
        self.Ntotal = np.sum(self.Ns)
        self.X = self.data[0].shape[2]
        self.Y = self.data[0].shape[3]
        self.cur_db = 0
        self.cur_cnt = 0
        # self.grid_49 = rz.grid_ab(40)
        self.grid_25 = rz.grid_ab(60)
        # self.grid_16 = rz.grid_ab(80)
        # self.grid_9 = rz.grid_ab(80)

    def reshape(self, bottom, top):
        top[0].reshape(self.N,1,self.X,self.Y)
        # top[1].reshape(self.N,49,self.X/self.SS,self.Y/self.SS)
        top[1].reshape(self.N,25,self.X/self.SS,self.Y/self.SS)
        # top[2].reshape(self.N,1,self.X/self.SS,self.Y/self.SS)
        # top[3].reshape(self.N,1,self.X/self.SS,self.Y/self.SS)
        # top[4].reshape(self.N,1,self.X/self.SS,self.Y/self.SS)
 
    def forward(self, bottom, top):
        # self.cur_db = 0
        # self.cur_cnt = 9952
        # print 'hello'

        # figure out which indices to sample from
        db_inds = {}
        cnt_inds = {}
        if(self.cur_cnt+self.N >= self.Ns[self.cur_db]):
            db_inds[0] = self.cur_db
            cnt_inds[0] = np.arange(self.cur_cnt,self.Ns[self.cur_db])

            # accumulate
            # print self.N
            # print self.Ns[self.cur_db]
            # print self.cur_cnt
            self.cur_cnt = self.N-(self.Ns[self.cur_db]-self.cur_cnt)

            self.cur_db = np.mod(self.cur_db+1,self.B)
            if(self.cur_cnt!=0):
                db_inds[1] = self.cur_db
                cnt_inds[1] = np.arange(0,self.cur_cnt)
        else:
            db_inds[0] = self.cur_db
            cnt_inds[0] = self.cur_cnt+np.arange(0,self.N)

            if(not self.TO_PROD_SAME): # assuming first file is bigger than a minibatch
                self.cur_cnt = self.cur_cnt+self.N # accumulate count
            else:
                self.cur_cnt = 0

        if(self.verbose):
            for cc in range(len(cnt_inds)):
                print('%s: %i: inds %i-%i from db %i'%(self.t.tocStr(),cc,cnt_inds[cc][0],cnt_inds[cc][-1],db_inds[cc]))

        # fill in data from the databases
        data_ab = np.zeros((self.N,2,self.X/self.SS,self.Y/self.SS),dtype='float32')

        cnt = 0
        for cc in range(len(cnt_inds)):
            inds = cnt+np.arange(0,cnt_inds[cc].size) # indices into data layer
            cnt = cnt+cnt_inds[cc].size
            db = db_inds[cc]

            # print cnt_inds[cc].flatten().size
            if(cnt_inds[cc].flatten().size==1):
                top[0].data[inds,:,:,:] = self.data[db][cnt_inds[cc],:,:,:][rz.na(),[0],:,:]
                # top[1].data[inds,:,:,:] = self.data[db][[cnt_inds[cc]],:,:,:][rz.na(),1:,::self.SS,::self.SS]
                data_ab[inds,:,:,:] = self.data[db][[cnt_inds[cc]],:,:,:][rz.na(),1:,::self.SS,::self.SS]
            else:
                top[0].data[inds,:,:,:] = self.data[db][cnt_inds[cc].flatten(),:,:,:][:,[0],:,:]
                data_ab[inds,:,:,:] = self.data[db][cnt_inds[cc].flatten(),:,:,:][:,1:,::self.SS,::self.SS]
                # top[1].data[inds,:,:,:] = self.data[db][cnt_inds[cc].flatten(),:,:,:][:,1:,::self.SS,::self.SS]

        data_ab = data_ab.transpose((0,2,3,1)).reshape((self.N*self.X/self.SS*self.Y/self.SS,2))
        # inds_enc = self.grid_49.encode_nearest_points(data_ab)
        # inds_enc = self.grid_49.encode_nearest_points(data_ab)
        # inds_enc = inds_enc.reshape((self.N,1,self.X/self.SS,self.Y/self.SS))
        # inds_enc = self.grid_49.encode_points(data_ab, returnMatrix=True)
        inds_enc = self.grid_25.encode_points(data_ab, returnMatrix=True)

        # inds_enc = inds_enc.reshape((self.N,self.X/self.SS,self.Y/self.SS,49)).transpose((0,3,1,2))
        inds_enc = inds_enc.reshape((self.N,self.X/self.SS,self.Y/self.SS,25)).transpose((0,3,1,2))
        top[1].data[...] = inds_enc.copy()

        # inds_enc = self.grid_25.encode_nearest_points(data_ab)
        # inds_enc = inds_enc.reshape((self.N,self.X/self.SS,self.Y/self.SS,49)).transpose((0,3,1,2))
        # inds_enc = inds_enc.reshape((self.N,1,self.X/self.SS,self.Y/self.SS))
        # top[2].data[...] = inds_enc

        # inds_enc = self.grid_16.encode_nearest_points(data_ab)
        # inds_enc = inds_enc.reshape((self.N,1,self.X/self.SS,self.Y/self.SS))
        # top[3].data[...] = inds_enc

        # inds_enc = self.grid_9.encode_nearest_points(data_ab)
        # inds_enc = inds_enc.reshape((self.N,1,self.X/self.SS,self.Y/self.SS))
        # top[4].data[...] = inds_enc

        # if(self.TO_NORM_AB):
            # top[1].data[...] = top[1].data[...]/110.
        # if(self.TO_NORM_AB):

        # top[1].data[...] = top[1].data[...]/110.

    def backward(self, top, propagate_down, bottom):
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)

# Load training data in LAB space, Quantize in square root polar space
# Should be in '/data/efros/rzhang/datasets/ILSVRC2012_processed'
#   /trn_data_lab_randord_224_%i.h5
#   binned together in 129 files with 10k files each
#   load self.N per batch
class ILSVRCTrnLabQuantCircDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 0:
            raise Exception("Data layer should not have inputs")

        self.verbose = True
        if(self.verbose):
            print('Loading ILSVRC training LAB data')
        # self.N = 96 # batch size
        self.N = 70 # batch size
        # self.B = 1 # total number of files
        self.B = 129 # total number of files
        self.verbose = True
        self.t = rz.Timer()
        # self.TO_NORM_AB = True # divide output by 110
        # self.TO_PROD_SAME = True # keep spitting out the first minibatch
        self.TO_PROD_SAME = False # keep spitting out the first minibatch
        self.load_overhead()
        self.SS = 8 # subsample on output

    def load_overhead(self):
        self.hdf5filedir = '/data/efros/rzhang/datasets/ILSVRC2012_processed'
        if(self.verbose):
            print('  Subdatabases found: %i'%self.B)
        self.hdf5filepaths = np.zeros(self.B,dtype='object')
        self.hdf5s = np.zeros(self.B,dtype='object')
        self.data = np.zeros(self.B,dtype='object')
        self.Ns = np.zeros(self.B,dtype='int')
        for bb in range(self.B):
            self.hdf5filepaths[bb] = os.path.join(self.hdf5filedir, 'trn_data_lab_randord_224_%i.h5'%(bb))
            self.hdf5s[bb] = rz.load_from_hdf5(self.hdf5filepaths[bb])
            self.data[bb] = self.hdf5s[bb]['data']
            self.Ns[bb] = self.data[bb].shape[0]
        self.Ntotal = np.sum(self.Ns)
        self.X = self.data[0].shape[2]
        self.Y = self.data[0].shape[3]
        self.cur_db = 0
        self.cur_cnt = 0
        self.pse = cq.PolarSqrtEncode()
        # self.grid_49 = rz.grid_ab(40)
        # self.grid_25 = rz.grid_ab(60)
        # self.grid_16 = rz.grid_ab(80)
        # self.grid_9 = rz.grid_ab(80)

    def reshape(self, bottom, top):
        top[0].reshape(self.N,1,self.X,self.Y)
        top[1].reshape(self.N,49,self.X/self.SS,self.Y/self.SS)
        # top[1].reshape(self.N,25,self.X/self.SS,self.Y/self.SS)
        # top[2].reshape(self.N,1,self.X/self.SS,self.Y/self.SS)
        # top[3].reshape(self.N,1,self.X/self.SS,self.Y/self.SS)
        # top[4].reshape(self.N,1,self.X/self.SS,self.Y/self.SS)
 
    def forward(self, bottom, top):
        # self.cur_db = 0
        # self.cur_cnt = 9952
        # print 'hello'

        # figure out which indices to sample from
        db_inds = {}
        cnt_inds = {}
        if(self.cur_cnt+self.N >= self.Ns[self.cur_db]):
            db_inds[0] = self.cur_db
            cnt_inds[0] = np.arange(self.cur_cnt,self.Ns[self.cur_db])

            # accumulate
            # print self.N
            # print self.Ns[self.cur_db]
            # print self.cur_cnt
            self.cur_cnt = self.N-(self.Ns[self.cur_db]-self.cur_cnt)

            self.cur_db = np.mod(self.cur_db+1,self.B)
            if(self.cur_cnt!=0):
                db_inds[1] = self.cur_db
                cnt_inds[1] = np.arange(0,self.cur_cnt)
        else:
            db_inds[0] = self.cur_db
            cnt_inds[0] = self.cur_cnt+np.arange(0,self.N)

            if(not self.TO_PROD_SAME): # assuming first file is bigger than a minibatch
                self.cur_cnt = self.cur_cnt+self.N # accumulate count
            else:
                self.cur_cnt = 0

        if(self.verbose):
            for cc in range(len(cnt_inds)):
                print('%s: %i: inds %i-%i from db %i'%(self.t.tocStr(),cc,cnt_inds[cc][0],cnt_inds[cc][-1],db_inds[cc]))

        # fill in data from the databases
        data_ab = np.zeros((self.N,2,self.X/self.SS,self.Y/self.SS),dtype='float32')

        cnt = 0
        for cc in range(len(cnt_inds)):
            inds = cnt+np.arange(0,cnt_inds[cc].size) # indices into data layer
            cnt = cnt+cnt_inds[cc].size
            db = db_inds[cc]

            # print cnt_inds[cc].flatten().size
            if(cnt_inds[cc].flatten().size==1):
                top[0].data[inds,:,:,:] = self.data[db][cnt_inds[cc],:,:,:][rz.na(),[0],:,:]
                # top[1].data[inds,:,:,:] = self.data[db][[cnt_inds[cc]],:,:,:][rz.na(),1:,::self.SS,::self.SS]
                data_ab[inds,:,:,:] = self.data[db][[cnt_inds[cc]],:,:,:][rz.na(),1:,::self.SS,::self.SS]
            else:
                top[0].data[inds,:,:,:] = self.data[db][cnt_inds[cc].flatten(),:,:,:][:,[0],:,:]
                data_ab[inds,:,:,:] = self.data[db][cnt_inds[cc].flatten(),:,:,:][:,1:,::self.SS,::self.SS]
                # top[1].data[inds,:,:,:] = self.data[db][cnt_inds[cc].flatten(),:,:,:][:,1:,::self.SS,::self.SS]

        # data_ab = data_ab.transpose((0,2,3,1)).reshape((self.N*self.X/self.SS*self.Y/self.SS,2))
        # inds_enc = self.grid_49.encode_nearest_points(data_ab)
        # inds_enc = self.grid_49.encode_nearest_points(data_ab)
        # inds_enc = inds_enc.reshape((self.N,1,self.X/self.SS,self.Y/self.SS))
        # inds_enc = self.grid_49.encode_points(data_ab, returnMatrix=True)
        # inds_enc = self.grid_25.encode_points(data_ab, returnMatrix=True)

        # inds_enc = inds_enc.reshape((self.N,self.X/self.SS,self.Y/self.SS,49)).transpose((0,3,1,2))
        # inds_enc = inds_enc.reshape((self.N,self.X/self.SS,self.Y/self.SS,25)).transpose((0,3,1,2))

        # top[1].data[...] = inds_enc.copy()

        top[1].data[...] = self.pse.encode_points_mtx_nd(data_ab,axis=1)

        # inds_enc = self.grid_25.encode_nearest_points(data_ab)
        # inds_enc = inds_enc.reshape((self.N,self.X/self.SS,self.Y/self.SS,49)).transpose((0,3,1,2))
        # inds_enc = inds_enc.reshape((self.N,1,self.X/self.SS,self.Y/self.SS))
        # top[2].data[...] = inds_enc

        # inds_enc = self.grid_16.encode_nearest_points(data_ab)
        # inds_enc = inds_enc.reshape((self.N,1,self.X/self.SS,self.Y/self.SS))
        # top[3].data[...] = inds_enc

        # inds_enc = self.grid_9.encode_nearest_points(data_ab)
        # inds_enc = inds_enc.reshape((self.N,1,self.X/self.SS,self.Y/self.SS))
        # top[4].data[...] = inds_enc

        # if(self.TO_NORM_AB):
            # top[1].data[...] = top[1].data[...]/110.
        # if(self.TO_NORM_AB):

        # top[1].data[...] = top[1].data[...]/110.

    def backward(self, top, propagate_down, bottom):
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)


# Load training data in LAB space, Quantize in using nearest neighbors and RBF kernel
# Should be in '/data/efros/rzhang/datasets/ILSVRC2012_processed'
#   /trn_data_lab_randord_224_%i.h5
#   binned together in 129 files with 10k files each
#   load self.N per batch
class ILSVRCTrnLabQuantNNDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 0:
            raise Exception("Data layer should not have inputs")

        self.verbose = True
        if(self.verbose):
            print('Loading ILSVRC training LAB data')
        # self.N = 96 # batch size
        self.N = 68 # batch size
        self.B = 129 # total number of files
        self.verbose = True
        self.t = rz.Timer()
        # self.TO_PROD_SAME = True # keep spitting out the first minibatch
        self.TO_PROD_SAME = False # keep spitting out the first minibatch
        self.load_overhead()
        self.SS = 8 # subsample on output

    def load_overhead(self):
        self.hdf5filedir = '/data/efros/rzhang/datasets/ILSVRC2012_processed'
        if(self.verbose):
            print('  Subdatabases found: %i'%self.B)
        self.hdf5filepaths = np.zeros(self.B,dtype='object')
        self.hdf5s = np.zeros(self.B,dtype='object')
        self.data = np.zeros(self.B,dtype='object')
        self.Ns = np.zeros(self.B,dtype='int')
        for bb in range(self.B):
            self.hdf5filepaths[bb] = os.path.join(self.hdf5filedir, 'trn_data_lab_randord_224_%i.h5'%(bb))
            self.hdf5s[bb] = rz.load_from_hdf5(self.hdf5filepaths[bb])
            self.data[bb] = self.hdf5s[bb]['data']
            self.Ns[bb] = self.data[bb].shape[0]
        self.Ntotal = np.sum(self.Ns)
        self.X = self.data[0].shape[2]
        self.Y = self.data[0].shape[3]
        self.cur_db = 0
        self.cur_cnt = 0
        self.NN = 15 # nearest neighbors
        self.sigma = 8 # sigma
        self.nnenc = cq.NNEncode(self.NN,self.sigma)

    def reshape(self, bottom, top):
        top[0].reshape(self.N,1,self.X,self.Y)
        top[1].reshape(self.N,150,self.X/self.SS,self.Y/self.SS)
        # top[1].reshape(self.N,1,self.X/self.SS,self.Y/self.SS)
 
    def forward(self, bottom, top):
        # self.cur_db = 60

        # figure out which indices to sample from
        db_inds = {}
        cnt_inds = {}
        if(self.cur_cnt+self.N >= self.Ns[self.cur_db]):
            db_inds[0] = self.cur_db
            cnt_inds[0] = np.arange(self.cur_cnt,self.Ns[self.cur_db])
            self.cur_cnt = self.N-(self.Ns[self.cur_db]-self.cur_cnt)
            self.cur_db = np.mod(self.cur_db+1,self.B)
            if(self.cur_cnt!=0):
                db_inds[1] = self.cur_db
                cnt_inds[1] = np.arange(0,self.cur_cnt)
        else:
            db_inds[0] = self.cur_db
            cnt_inds[0] = self.cur_cnt+np.arange(0,self.N)
            if(not self.TO_PROD_SAME): # assuming first file is bigger than a minibatch
                self.cur_cnt = self.cur_cnt+self.N # accumulate count
            else:
                self.cur_cnt = 0 # do not accumulate, just produce same thing

        if(self.verbose):
            for cc in range(len(cnt_inds)):
                print('%s: %i: inds %i-%i from db %i'%(self.t.tocStr(),cc,cnt_inds[cc][0],cnt_inds[cc][-1],db_inds[cc]))

        # fill in data from the databases
        data_ab = np.zeros((self.N,2,self.X/self.SS,self.Y/self.SS),dtype='float32')

        cnt = 0
        for cc in range(len(cnt_inds)):
            inds = cnt+np.arange(0,cnt_inds[cc].size) # indices into data layer
            cnt = cnt+cnt_inds[cc].size
            db = db_inds[cc]

            # print cnt_inds[cc].flatten().size
            if(cnt_inds[cc].flatten().size==1):
                top[0].data[inds,:,:,:] = self.data[db][cnt_inds[cc],:,:,:][rz.na(),[0],:,:]
                data_ab[inds,:,:,:] = self.data[db][[cnt_inds[cc]],:,:,:][rz.na(),1:,::self.SS,::self.SS]
            else:
                top[0].data[inds,:,:,:] = self.data[db][cnt_inds[cc].flatten(),:,:,:][:,[0],:,:]
                data_ab[inds,:,:,:] = self.data[db][cnt_inds[cc].flatten(),:,:,:][:,1:,::self.SS,::self.SS]

        top[1].data[...] = self.nnenc.encode_points_mtx_nd(data_ab,axis=1)
        # top[1].data[...] = np.argmax(self.nnenc.encode_points_mtx_nd(data_ab,axis=1),axis=1)[:,rz.na(),:,:]

        if(np.random.rand()>.5): # flip
            top[0].data[:,:,:,:] = top[0].data[:,:,::-1,:]
            top[1].data[:,:,:,:] = top[1].data[:,:,::-1,:]
            # print 'Flipped!'

    def backward(self, top, propagate_down, bottom):
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)
