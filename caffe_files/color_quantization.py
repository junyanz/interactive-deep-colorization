#!/usr/bin/env python
## Richard Zhang / 2016.02.09
# Quantize colors in AB space

# %load_ext autoreload
# %autoreload 2

import os
import numpy as np
import scipy as sp
import caffe
import matplotlib.pyplot as plt
import time
import rz_fcns as rz
# import open_caffe_network as ocn
# import ilsvrc12_loader as ill
from skimage import color
from mpl_toolkits.mplot3d import Axes3D as ax3
from IPython.core.debugger import Pdb as pdb
import sklearn.neighbors as nn
# import ilsvrc12_loader as ill
import scipy
import caffe

class GaussianPyramid():
    def __init__(self,Lmax=5,Lmin=2):
        self.Lmax = Lmax
        self.Lmin = Lmin
        self.Ldelt = Lmax-Lmin
        self.img_xform = rz.img_data_lab_transformer()

    def pyramid_data_lab(self,data_lab):
        # INPUTS
        #   data_lab        Nx3xXxY
        # OUTPUTS
        #   py_data_lab     Ldelt list of Nx3xXlxYl   pyramid of images
        N = data_lab.shape[0]
        X = data_lab.shape[2]
        Y = data_lab.shape[3]

        # transform image
        imgs_rgb = self.img_xform.data2img(data_lab)
        py_img_rgb = self.pyramid_img_rgb(imgs_rgb)

        # initialize pyramids
        py_data_lab = {}
        for ll in range(self.Ldelt):
            py_data_lab[ll] = self.img_xform.img2data(py_img_rgb[ll])
        
        return py_data_lab

    def pyramid_img_rgb(self,imgs_rgb):
        # INPUTS
        #   imgs_lab        Nx3xXxY
        # OUTPUTS
        #   py_img_lab     Ldelt list of Nx3xXlxYl   pyramid of images
        py_img_rgb = {}
        for ll in range(self.Ldelt):
            py_img_rgb[ll] = np.zeros((Y/2**(ll+self.Lmin),X/2**(ll+self.Lmin),3,N),dtype='float32')

        # create image pyramids
        for ii in range(N):
            pyramid = tuple(pg(imgs_rgb[:,:,:,ii],max_layer=self.Lmax-1,downscale=2,sigma=None,order=1,mode='reflect',cval=0))
            pyramid = pyramid[self.Lmin:]
            for ll in range(self.Ldelt):
                py_img_rgb[ll][:,:,:,ii] = pyramid[ll]

        return py_img_rgb

def ab_encode_decode(genc,ab_in,isDistr=False,toSoftmax=False,axis=0,procMean=True,procMode4Hot=True,procMode1Hot=True,procMed1Hot=True):
	''' Drive the individual encoding/decoding methods:
	(a) full (MEAN) (b) 4hot/1hot (MODE) (c) 1hot (MEDIAN) encoding/decoding
	# INPUTS
	# 	genc 		GridEncode object
	# 	l_in 	 	1xXxY
	# 	ab_in 		QxXxY or 2xXxY
	# 	img_xform 	image transformer
	# 	isDistr		bool 		[False] real value if False, distribution over quantized space if True
	# 	toSoftmax 	bool 		[False] push through a softmax if True
	# 	axis 		integer 	axis of ab values
	# 	US 			integer 	upsampling factor
	# OUTPUTS
	#  	ab 			dict or encoded and decoded distributions
	# 	entr 	 	dict 	
	# 	lab 		dict of LAB data, 3xXxY
	# 	rgb 		dict of RGB images, YxXx3 '''

	ab_enc = {} # ab encoding and decodings
	ab_dec = {}
	entr = {} # Entropy
	if(isDistr):
		ab_enc['full'] = ab_in.copy()
	else:
		ab_enc['full'] = genc.encode_nn_mtx_nd(ab_in,axis=axis)
	if(toSoftmax):
		ab_enc['full'] = rz.softmax_mtx_nd(ab_enc['full'],axis=axis)
	ab_enc['full_full'] = genc.enc_full_grid_mtx_nd(ab_enc['full'],axis=axis)
	ab_enc['full_full'] = rz.reshape_single_axis(ab_enc['full_full'],genc.grid.A,genc.grid.B,axis=axis)

	if(procMean):
		ab_dec['full'] = genc.decode_nn_mtx_nd(ab_enc['full'],axis=axis)
		entr['full'] = entropy_mtx_nd(ab_enc['full'],axis=axis)
	if(procMode4Hot):
		(ab_dec['4hot'],ab_enc['4hot']) = genc.decode_4hot_mtx_nd(ab_enc['full'],axis=axis,returnEncode=True)
		entr['4hot'] = entropy_mtx_nd(ab_enc['4hot'],axis=axis)
	if(procMode1Hot):
		(ab_dec['1hot'],ab_enc['1hot']) = genc.decode_1hot_mtx_nd(ab_enc['full'],axis=axis,returnEncode=True)
		entr['1hot'] = entropy_mtx_nd(ab_enc['1hot'],axis=axis)
	if(procMed1Hot):
		(ab_dec['1hotmed'],ab_enc['1hotmed']) = genc.decode_med_mtx_nd(ab_enc['full'],axis=axis,returnEncode=True)
		entr['1hotmed'] = entropy_mtx_nd(ab_enc['1hotmed'],axis=axis)

	return (ab_enc,ab_dec,entr)

def upsample_grids(grids,US=1,X=224,axes=0,useUS=True,order=1):
	''' Upsample grids
	# INPUTS
	 	grids 		dict of nd arrays or an nd array
	 	US 			integer that is upsampling factor
	 	axes 		indices to upsample over
	 	X 			amount to upsample to
	 	useUS 		bool, [True] uses upsampling factor, [False] uses the X
	 	order 		
	 OUTPUTS
	 	out_grids 	 '''
	axes = np.array((axes))
	if(type(grids)==dict):
		in_dict = True
	else:
		tmp = grids.copy()
		grids = {}
		grids['a'] = tmp
		in_dict = False

	X = np.array(X)
	if(X.size==1):
		X = X+np.zeros_like(axes)

	for key in grids.keys():
		US_array = np.ones(grids[key].ndim)
		for (aa,ax) in enumerate(axes.flatten()):
			if(useUS):
				US_array[ax] = US
			else:
				US_array[ax] = 1.*X[aa]/grids[key].shape[ax]
		grids[key] = sp.ndimage.interpolation.zoom(grids[key],US_array,order=order) # ,mode='reflect'

	if(in_dict):
		return grids
	else:
		return grids['a']

def data_l_abs_to_imgs(data_ab,data_l,img_xform):
	''' Concatenate dict of ab channels to a single luminance channel
	INPUTS
	 	data_ab 		dict of NxCxXxY
		data_l 			NxCxXxY
		img_xform 		image transform object
	'''
	# ***** LAB,RGB Images *****
	if(data_l.ndim==3):
		data_l = data_l[rz.na(),:,:,:]
	(X,Y) = data_l.shape[2:]
	N = data_l.shape[0]

	lab = {}
	rgb = {}
	for key in data_ab.keys():
		lab[key] = np.zeros((N,3,X,Y))
		for nn in range(N):
			if(data_ab[key].ndim==3):
				data_ab_exp = data_ab[key][rz.na(),:,:,:]
			else:
				data_ab_exp = data_ab[key].copy()
			(C,X0,Y0) = data_ab_exp.shape[1:]
			if(X0==X and Y0==Y):
				ab_us = data_ab_exp[nn,:,:,:]
			else:
				ab_us = caffe.io.resize(data_ab_exp[nn,:,:,:].astype('float64'),(C,X,Y)).astype('float32')
				# grids[key] = sp.ndimage.interpolation.zoom(grids[key],US_array,order=1) # ,mode='reflect'
			lab[key][nn,1:,:,:] = ab_us
		lab[key][:,0,:,:] = data_l[:,0,:,:]
		rgb[key] = img_xform.data2img(lab[key])

	return (lab,rgb)

def load_grid_pts(grid_inc=10,GRIDDIR='/home/eecs/rich.zhang/src/projects/cross_domain/ab_grid_10'):
	# load ab points in hull
	# GRIDDIR = 
	grid_pts = {}
	grid_pts['pts_grid'] = np.load(os.path.join(GRIDDIR,'pts_grid.npy'))
	grid_pts['pts_masked'] = np.load(os.path.join(GRIDDIR,'pts_in_hull.npy'))
	grid_pts['mask'] = np.load(os.path.join(GRIDDIR,'in_hull.npy'))
	grid_pts['grid'] = rz.grid_ab(grid_inc)

	return grid_pts

def expand_axis_mask(pts,grid_mask,axis=1,returnFlat=False):
	''' Expand a single axis by a mask.
	INPUTS
		pts 		N0xN1x...xNn 	n-dimensional matrix
	 	grid_mask 	M bool vector
	 	axis 		integer 		axis to expand
	 	returnFlat 	boolean 		if True, return a PxM vector
	 								if False, return a N0xN1x...xNn, but with Naxis replaced with M '''
	AB = grid_mask.size
	pts_flt = rz.flatten_nd_array(pts,axis=axis)
	P = pts_flt.shape[0]
	pts_full_flt = np.zeros((P,AB),dtype='float32')
	pts_full_flt[:,grid_mask] = pts_flt
	if(returnFlat):
		return pts_full_flt
	else:
		return rz.unflatten_2d_array(pts_full_flt,pts,axis=axis)

def is_data_lab_gray(data_lab,thresh=5):
	# INPUTS
	# 	data_lab 	Nx3xXxY 	lab data
	# OUTPUTS
	# 	mask 		N bool 		return true if image is in grayscale

	if(data_lab.shape[1]==3):
		return np.sum(np.sum(np.sum((np.abs(data_lab[:,1:,:,:]) > thresh),axis=1),axis=1),axis=1)==0
	elif(data_lab.shape[1]==2):
		return np.sum(np.sum(np.sum((np.abs(data_lab) > thresh),axis=1),axis=1),axis=1)==0

def is_data_lab_monochrome(data_lab,thresh=5):
	# INPUTS
	# 	data_lab 	Nx3xXxY 	lab data
	# OUTPUTS
	# 	mask 		N bool 		return true if image is in grayscale

	max_variation = np.max(np.max(np.max(data_lab[:,1:,:,:],axis=1),axis=1),axis=1) - np.min(np.min(np.min(data_lab[:,1:,:,:],axis=1),axis=1),axis=1)
	return max_variation < thresh

	# if(data_lab.shape[1]==3):
	# 	return np.sum(np.sum(np.sum((np.abs(data_lab[:,1:,:,:]) > 5),axis=1),axis=1),axis=1)==0
	# elif(data_lab.shape[1]==2):
	# 	return np.sum(np.sum(np.sum((np.abs(data_lab) > 5),axis=1),axis=1),axis=1)==0

def collect_pixel_distr(HDF5_FILENAME,ab_inc=2,SS=8,OFF=0,N=1000,toNorm=True,filtGray=True):
	# Collects distribution of pixels across validation set
	# INPUTS
	# 	HDF_FILENAME 	string 		hdf file with Nx3xXxY LAB images
	# 	ab_inc 			integer 	grid granularity
	# 	SS 				integer 	subsample pixels
	# 	OFF 			integer 	offset
	# 	N 				integer 	number of images to load
	# 	toNorm			boolean 	normalize output so it is a PMF
	# OUTPUTS
	# 	hist_ab_cnt_accum  	grid.a x grid.b array
	# 	grid 			grid object incremented at ab_inc

	val_data = rz.load_from_hdf5(HDF5_FILENAME)
	(X,Y) = val_data['data'][0,:,:,:].shape[1:]
	grid = rz.grid_ab(ab_inc)

	hist_ab_cnt_accum = np.zeros((grid.A,grid.B))
	b = rz.Batcher(N,100,update_interval=10)
	for bb in range(b.B):
		b_inds = OFF+b.increment()
		data_l = val_data['data'][b_inds,:,:,:][:,[0],::SS,::SS]
		data_ab = val_data['data'][b_inds,:,:,:][:,[1,2],::SS,::SS]
		if(filtGray):
			mask = is_data_lab_gray(data_ab)
			data_ab = data_ab[~mask,:,:,:]
			# print '  Filtered out %i grayscale images...'%np.sum(mask)

		data_ab_flt = rz.flatten_nd_array(data_ab,axis=1) # flatten
		hist_ab_cnt_accum+=np.histogram2d(data_ab_flt[:,0],data_ab_flt[:,1],bins=(grid.a_vals_edge,grid.b_vals_edge))[0]

		b.print_update()

	if(toNorm):
		hist_ab_cnt_accum = 1.*hist_ab_cnt_accum/np.sum(hist_ab_cnt_accum)
	return (hist_ab_cnt_accum,grid)

def collect_pixel_distr_pyramid(HDF5_FILENAME,gp,ab_inc=2,SS=8,OFF=0,N=1000,toNorm=True):
	# Collects distribution of pixels across validation set
	# INPUTS
	# 	HDF_FILENAME 	string 		hdf file with Nx3xXxY LAB images
	# 	gp 				GaussianPyramid object
	# 	ab_inc 			integer 	grid granularity
	# 	SS 				integer 	subsample pixels
	# 	OFF 			integer 	offset
	# 	N 				integer 	number of images to load
	# 	toNorm			boolean 	normalize output so it is a PMF

	val_data = rz.load_from_hdf5(HDF5_FILENAME)
	(X,Y) = val_data['data'][0,:,:,:].shape[1:]
	grid = rz.grid_ab(ab_inc)

	hist_ab_cnt_accum = np.zeros((grid.A,grid.B))
	b = rz.Batcher(N,100,update_interval=10)
	for bb in range(b.B):
		b_inds = OFF+b.increment()

		data_lab = val_data['data'][b_inds,:,:,:]
		gp.pyramid_img_lab(data_lab)

		data_ab_flt = rz.flatten_nd_array(data_ab,axis=1) # flatten
		hist_ab_cnt_accum+=np.histogram2d(data_ab_flt[:,0],data_ab_flt[:,1],bins=(grid.a_vals_edge,grid.b_vals_edge))[0]

		b.print_update()

	if(toNorm):
		hist_ab_cnt_accum = 1.*hist_ab_cnt_accum/np.sum(hist_ab_cnt_accum)
	return (hist_ab_cnt_accum,grid)

def entropy_mtx_nd(distr_nd,axis=1,eps=1e-10):
	distr_flt = rz.flatten_nd_array(distr_nd,axis=axis)
	entropy_flt = -np.sum(distr_flt*np.log10(distr_flt+eps),axis=1)[:,rz.na()]
	entropy_nd = rz.unflatten_2d_array(entropy_flt,distr_nd,axis=axis,squeeze=True)
	return entropy_nd

def int2kstr(num):
	return '%sk'%str(int(num/1000))

class GridEncode():
	def __init__(self,NN,sigma,grid_inc=10):
		# self.GRIDDIR = '/home/eecs/rich.zhang/data_rzhang/models/caffe/cross_domain/l_to_ab/2015_02_13_classification_nn_rbf_reggrid/'
		self.GRIDDIR = '/home/eecs/rich.zhang/src/projects/cross_domain/save/ab_grid_10'
		self.GRID_FULL_PATH = os.path.join(self.GRIDDIR,'pts_grid.npy')
		self.GRID_MASK_PATH = os.path.join(self.GRIDDIR,'in_hull.npy')
		self.GRID_HULL_PATH = os.path.join(self.GRIDDIR,'pts_in_hull.npy')

		self.grid_full = np.load(self.GRID_FULL_PATH) # all of the gridded points under consideration
		self.grid_mask = np.load(self.GRID_MASK_PATH) # mask in covex hull in AB space
		self.grid_hull = np.load(self.GRID_HULL_PATH) # gridded points in convex hull in AB space
		self.AB_hull = self.grid_hull.shape[0] # number of points in hull
		# self.AB_full = self.grdid_full.shape[0] # full grid
		# self.A_full = np.sqrt(self.AB_full) # 
		# self.B_full = np.sqrt(self.AB_full)

		# NN RBF Encoder object
		self.NN = NN # number of NNs to consider when encoding
		self.sigma = sigma # sigma when assigning weights using RBF kernel
		self.nnenc = NNEncode(self.NN,self.sigma,self.GRID_HULL_PATH)

		# Grid Encoder object
		self.grid = rz.grid_ab(grid_inc) # grid object

		# 1-hot encoder
		self.nn1enc = NNEncode(1,self.sigma,self.GRID_HULL_PATH)

	def encode_nn_mtx_nd(self,pts_nd,axis=1):
		''' Encode by
		# 	(1) finding nearest neighbors
		# 	(2) weighting nns by RBF kernel
		# 	(3) normalizing '''
		return self.nnenc.encode_points_mtx_nd(pts_nd,axis=axis)

	def encode_4hot_mtx_nd(self,pts_nd,axis=1):
		''' Encode by
		# 	(1) finding 4 corners around grid
		# 	(2) weighting as a linear interpolation
		# 	(3) masking out points which are in hull
		# 	(4) re-normalizing (in case some points are on border) '''

		pts_flt = rz.flatten_nd_array(pts_nd,axis=axis)
		pts_enc_full_flt = self.grid.encode_points_mtx_nd(pts_nd,axis=axis,returnFlat=True)		
		pts_enc_flt = pts_enc_full_flt[:,self.grid_mask]
		pts_enc_flt = pts_enc_flt/np.sum(pts_enc_flt,axis=1)[:,rz.na()]
		return rz.unflatten_2d_array(pts_enc_flt,pts_nd,axis=axis)

	def encode_1hot_mtx_nd(self,pts_nd,axis=1):
		''' Encode by
		# 	(1) finding 1-NN '''
		return self.nn1enc.encode_points_mtx_nd(pts_nd,axis=axis)

	def decode_nn_mtx_nd(self,pts_nd,axis=1):
		''' Decode by finding MEAN
		# 	(1) taking weighted average over all positions '''
		return self.nnenc.decode_points_mtx_nd(pts_nd,axis=axis)

	def decode_4hot_mtx_nd(self,pts_nd,axis=1,returnEncode=False):
		''' Decode by finding MODE
		# 	(0) converting classification of points in hull into all points in grid
		# 	(1) returning position of hottest 2x2 region '''

		# t = rz.Timer()
		pts_full_flt = self.enc_full_grid_mtx_nd(pts_nd,axis=axis,returnFlat=True)
		# print ' %s'%t.tocStr()

		pts_full_4hot_flt = nd_argmax_4hot_conv22(pts_full_flt,self.grid.A,self.grid.B,axis=1)
		# print ' %s'%t.tocStr()

		pts_dec_full_4hot_flt = self.grid.decode_points(pts_full_4hot_flt)
		# print ' %s'%t.tocStr()
		pts_dec_full_4hot = rz.unflatten_2d_array(pts_dec_full_4hot_flt,pts_nd,axis=axis)
		# print ' %s'%t.tocStr()
		if(returnEncode):
			pts_4hot_flt = pts_full_4hot_flt[:,self.grid_mask]
			# print ' %s'%t.tocStr()
			pts_full_4hot = rz.unflatten_2d_array(pts_4hot_flt,pts_nd,axis=axis)
			# print ' %s'%t.tocStr()
			return (pts_dec_full_4hot,pts_full_4hot)
		else:
			return pts_dec_full_4hot

	def decode_1hot_mtx_nd(self,pts_nd,axis=1,returnEncode=False):
		''' Decode by finding MODE
		# 	(1) returning position of hottest entry '''

		pts_1hot_nd = nd_argmax_1hot(pts_nd,axis=axis)
		pts_dec_nd = self.nn1enc.decode_points_mtx_nd(pts_1hot_nd,axis=axis)
		if(returnEncode):
			return (pts_dec_nd,pts_1hot_nd)
		else:
			return pts_dec_nd

	def decode_med_mtx_nd(self,pts_nd,axis=1,returnEncode=False):
		''' Encode by MEDIAN '''

		pts_full_flt = self.enc_full_grid_mtx_nd(pts_nd,axis=axis,returnFlat=True)
		med_enc_full_flt = nd_argmax_med_grid(pts_full_flt,self.grid.A,self.grid.B,axis=1)
		
		med_enc_flt = med_enc_full_flt[:,self.grid_mask]
		med_enc_nd = rz.unflatten_2d_array(med_enc_flt, pts_nd, axis=axis)

		# print med_enc_flt.shape
		# print self.grid.ab_grid_flat.shape
		pts_dec_flt = self.decode_nn_mtx_nd(med_enc_flt,axis=1)
		pts_dec_nd = rz.unflatten_2d_array(pts_dec_flt,pts_nd,axis=axis)

		if(returnEncode):
			return (pts_dec_nd,med_enc_nd)
		else:
			return pts_dec_nd

	def enc_full_grid_mtx_nd(self,pts_enc,axis=1,returnFlat=False,returnGrid=False):
		''' Expand a distribution into full distribution over non-legit ab values
		INPUTS
		 	pts_enc 		NxCxXxY		encoded nd mtx (typically)
		 	axis 			integer
		 	returnFlat 		bool
		 	returnGrid 		bool 		expand dimension to AxB
		OUTPUTS
		 	pts_full 		if returnGrid 		NXYxAxB or NxAxBxXxY
		 	 				if not returnGrid 	NXYxC_full or NxC_fullxXxY '''
		pts_flt = rz.flatten_nd_array(pts_enc,axis=axis)
		P = pts_flt.shape[0]

		pts_full_flt = np.zeros((P,self.grid.AB),dtype='float32')
		pts_full_flt[:,self.grid_mask] = pts_flt
		if(returnFlat):
			ret_mtx = pts_full_flt
			ret_axis = 1
		else:
			ret_mtx = rz.unflatten_2d_array(pts_full_flt,pts_enc,axis=axis)
			ret_axis = axis

		if(returnGrid):
			return rz.reshape_single_axis(ret_mtx,self.grid.A,self.grid.B,axis=ret_axis)
		else:
			return ret_mtx

class NNEncode():
	# Encode points as a linear combination of unordered points
	# using NN search and RBF kernel
	def __init__(self,NN,sigma,km_filepath='/home/eecs/rich.zhang/data_rzhang/models/caffe/cross_domain/l_to_ab/2015_02_12_classification_nn_rbf/cc_k150.npy',cc=-1):
		if(rz.check_value(cc,-1)):
			self.cc = np.load(km_filepath)
		else:
			self.cc = cc
		self.K = self.cc.shape[0]
		# self.NN = NN
		self.NN = int(NN)
		self.sigma = sigma
		self.nbrs = nn.NearestNeighbors(n_neighbors=NN, algorithm='ball_tree').fit(self.cc)

	def encode_points_mtx_nd(self,pts_nd,axis=1,returnSparse=False):
		t = rz.Timer()
		pts_flt = rz.flatten_nd_array(pts_nd,axis=axis)
		P = pts_flt.shape[0]

		(dists,inds) = self.nbrs.kneighbors(pts_flt)

		pts_enc_flt = np.zeros((P,self.K))
		wts = np.exp(-dists**2/(2*self.sigma**2))
		wts = wts/np.sum(wts,axis=1)[:,rz.na()]

		pts_enc_flt[np.arange(0,P,dtype='int')[:,rz.na()],inds] = wts
		pts_enc_nd = rz.unflatten_2d_array(pts_enc_flt,pts_nd,axis=axis)

		return pts_enc_nd

	def decode_points_mtx_nd(self,pts_enc_nd,axis=1):
		pts_enc_flt = rz.flatten_nd_array(pts_enc_nd,axis=axis)
		pts_dec_flt = np.dot(pts_enc_flt,self.cc)
		pts_dec_nd = rz.unflatten_2d_array(pts_dec_flt,pts_enc_nd,axis=axis)
		return pts_dec_nd

	def decode_1hot_mtx_nd(self,pts_enc_nd,axis=1,returnEncode=False):
		pts_1hot_nd = nd_argmax_1hot(pts_enc_nd,axis=axis)
		pts_dec_nd = self.decode_points_mtx_nd(pts_1hot_nd,axis=axis)
		if(returnEncode):
			return (pts_dec_nd,pts_1hot_nd)
		else:
			return pts_dec_nd


class PolarSqrtEncode():
	def __init__(self):
		self.rs = np.arange(0,14,2.)**2
		self.thetas = np.arange(-180,210,45.)
		self.grid = rz.Grid2DEncoder(0.,14.,2.,-180.,210.,60.) # grid in squareroot space
		self.grid_pts = sqrtpolar2ab(self.grid.ab_grid_flat)

	def encode_points_mtx_nd(self,pts_nd,axis=1):
		# INPUTS
		# 	pts_nd 	nd points in ab space
		# 	axis 	axis containing ab

		# self.grid()
		pts_flt = rz.flatten_nd_array(pts_nd,axis=axis)
		# print pts_flt.shape

		pts_flt_polar_sqrt = np.zeros_like(pts_flt)
		pts_flt_polar_sqrt[:,0] = np.sqrt(np.sqrt(np.sum(pts_flt**2,axis=1)))
		pts_flt_polar_sqrt[:,1] = np.arctan2(pts_flt[:,0],pts_flt[:,1])*180/np.pi
		# print pts_flt_polar_sqrt.shape
		# print pts_flt_polar_sqrt

		pts_flt_enc = self.grid.encode_points(pts_flt_polar_sqrt, returnMatrix=True)
		# print pts_flt_enc.shape

		pts_enc_nd = rz.unflatten_2d_array(pts_flt_enc, pts_nd, axis=axis)

		return pts_enc_nd

	def decode_points_mtx_nd(self,pts_enc_nd,axis=1):
		# INPUTS
		# 	pts_enc_nd 		encoded nd points in ab space
		# 	axis 			axis containing ab

		pts_nd_polar_sqrt = self.grid.decode_points_mtx_nd(pts_enc_nd, axis=axis)
		pts_flt_polar_sqrt = rz.flatten_nd_array(pts_nd_polar_sqrt,axis=axis)

		pts_flt_polar = pts_flt_polar_sqrt.copy()
		pts_flt_polar[:,0] = pts_flt_polar[:,0]**2
		pts_flt_polar[:,1] = pts_flt_polar[:,1]*np.pi/180

		pts_flt = np.zeros_like(pts_flt_polar)
		pts_flt[:,0] = pts_flt_polar[:,0]*np.sin(pts_flt_polar[:,1])
		pts_flt[:,1] = pts_flt_polar[:,0]*np.cos(pts_flt_polar[:,1])

		pts_nd = rz.unflatten_2d_array(pts_flt,pts_enc_nd,axis=axis)
		return pts_nd

def nd_argmax_1hot_4d(pts_enc_4d,axis=1):
	# Compute highest index along axis, return 1-hot encoding along that axis
	# INPUTS
	# 	pts_enc_4d 			NxCxXxY (typically)
	# 	axis 				integer which contains channel dimension
	# OUTPUTS
	# 	pts_enc_1hot_4d 	NxCxXxY (typically) with 1 non-zero values along each channel

	SHP = np.array(pts_enc_4d.shape) # shape
	nax = np.setdiff1d(np.arange(0,4),np.array((axis))) # non axis indices
	ogrid = np.ogrid[:SHP[nax[0]],:SHP[nax[1]],:SHP[nax[2]]] # non axis grid indices

	max_inds = np.argmax(pts_enc_4d,axis=axis) # max over specificed axis

	pts_enc_1hot_4d = np.zeros_like(pts_enc_4d)
	if(axis==0):
		pts_enc_1hot_4d[max_inds,ogrid[0],ogrid[1],ogrid[2]] = True
	elif(axis==1):
		pts_enc_1hot_4d[ogrid[0],max_inds,ogrid[1],ogrid[2]] = True
	elif(axis==2):
		pts_enc_1hot_4d[ogrid[0],ogrid[1],max_inds,ogrid[2]] = True
	elif(axis==3):
		pts_enc_1hot_4d[ogrid[0],ogrid[1],ogrid[2],max_inds] = True
	return pts_enc_1hot_4d

def nd_argmax_med_grid(pts_enc_nd,A,B,axis=1):
	# Compute index for median, assuming distribution is over a grid
	# (1) Reshape axis to be 2d
	# (2) Run cum-sum in those 2 dimensions
	# (3) Find median point in those 2 dimensions
	# INPUTS
	# 	pts_enc_nd 		NxCxXxY (typically), or any nd array
	# 	A 				integer 	AxB=C
	# 	B 				integer 	
	# 	axis 			integer 	which contains channel C dimension
	# OUTPUTS
	# 	med_enc_nd 		NxCxXxY

	pts_enc_flt = rz.flatten_nd_array(pts_enc_nd,axis=axis)
	NXY = pts_enc_flt.shape[0]
	pts_enc_flt = pts_enc_flt.reshape((NXY,A,B))

	# marginalize out in either dimension
	pts_enc_full_flt_marg0 = np.sum(pts_enc_flt,axis=2)
	pts_enc_full_flt_marg1 = np.sum(pts_enc_flt,axis=1)

	med_inds0 = np.argmin(np.abs(np.cumsum(pts_enc_full_flt_marg0,axis=1)-.5),axis=1)
	med_inds1 = np.argmin(np.abs(np.cumsum(pts_enc_full_flt_marg1,axis=1)-.5),axis=1)

	# plt.hist(np.min(np.abs(np.cumsum(pts_enc_full_flt_marg0,axis=1)-.5),axis=1))
	# plt.show()

	med_subs = np.concatenate((med_inds0[:,rz.na()],med_inds1[:,rz.na()]),axis=1)
	med_inds = rz.sub2ind2(med_subs,(A,B))

	med_enc_flt = np.zeros((NXY,A*B),dtype='float32')
	med_enc_flt[np.arange(0,NXY),med_inds] = 1

	med_enc_nd = rz.unflatten_2d_array(med_enc_flt,pts_enc_nd,axis=axis)

	return med_enc_nd

def nd_argmax_4hot_conv22(pts_enc_nd,A,B,axis=1):
	# Compute highest index along axis, return 4-hot encoding along that axis
	# run a 2x2 convolution to figure out how which 4 indices to keep
	# (1) Reshape axis to be 2d
	# (2) Find maximum location
	# (3) Map to the 4 indices in that location
	# INPUTS
	# 	pts_enc_nd 		NxCxXxY (typically), or any nd array
	# 	A 				integer 	AxB=C
	# 	B 				integer 	
	# 	axis 			integer 	which contains channel C dimension
	# OUTPUTS
	# 	pts_enc_1hot_4d 	NxCxXxY (typically) with 1 non-zero values along each channel

	# flatten into NXYxC
	# reshape into NXYxAxB

	# t = rz.Timer()
	pts_enc_flt = rz.flatten_nd_array(pts_enc_nd,axis=axis)
	NXY = pts_enc_flt.shape[0]
	pts_enc_flt = pts_enc_flt.reshape((NXY,A,B))
	# print '  %s'%t.tocStr()

	# run 2x2 convolution
	pts_enc_flt_conv22 = pts_enc_flt[:,:-1,:-1] + pts_enc_flt[:,1:,:-1] + pts_enc_flt[:,:-1,1:] + pts_enc_flt[:,1:,1:]
	pts_enc_flt_conv22 = pts_enc_flt_conv22.reshape((NXY,(A-1)*(B-1)))
	high_ind = np.argmax(pts_enc_flt_conv22,axis=1)
	high_sub = rz.ind2sub2(high_ind,(A-1,B-1))
	# print '  %s'%t.tocStr()

	# find top location
	ogrid = np.ogrid[:NXY,:A,:B]

	mask = np.zeros((NXY,A,B),dtype='bool')
	mask[ogrid[0][:,0,0],high_sub[:,0],high_sub[:,1]] = True
	mask[ogrid[0][:,0,0],high_sub[:,0]+1,high_sub[:,1]] = True
	mask[ogrid[0][:,0,0],high_sub[:,0],high_sub[:,1]+1] = True
	mask[ogrid[0][:,0,0],high_sub[:,0]+1,high_sub[:,1]+1] = True
	# print '  %s'%t.tocStr()

	# mask & renormalize
	pts_enc_flt = (pts_enc_flt * mask).reshape((NXY,A*B))
	pts_enc_flt = pts_enc_flt / np.sum(pts_enc_flt,axis=1)[:,rz.na()]
	pts_enc_flt = rz.unflatten_2d_array(pts_enc_flt, pts_enc_nd, axis=axis)
	# print '  %s'%t.tocStr()

	return pts_enc_flt

def nd_argmax_1hot(pts_enc_nd,axis=1):
	# Compute highest index along 'axis', return 1 hot encoding of that axis
	# INPUTS
	# 	pts_enc_nd 	nd array
	# 	axis 			integer to perform argmax over
	pts_enc_flt = rz.flatten_nd_array(pts_enc_nd,axis=axis)
	N = pts_enc_flt.shape[0]

	pts_enc_1hot_flt = np.zeros_like(pts_enc_flt)
	max_inds = np.argmax(pts_enc_flt,axis=1)
	pts_enc_1hot_flt[np.arange(0,N),max_inds] = True

	pts_enc_1hot_nd = rz.unflatten_2d_array(pts_enc_1hot_flt,pts_enc_nd,axis=axis)

	return pts_enc_1hot_nd		

def nd_argmax_1hot(pts_enc_nd,axis=1):
	# Run 2x2 convolution, pick highest index, return 
	# INPUTS
	# 	pts_enc_nd 	nd array
	# 	axis 			integer to perform argmax over
	pts_enc_flt = rz.flatten_nd_array(pts_enc_nd,axis=axis)
	N = pts_enc_flt.shape[0]

	pts_enc_1hot_flt = np.zeros_like(pts_enc_flt)
	max_inds = np.argmax(pts_enc_flt,axis=1)
	pts_enc_1hot_flt[np.arange(0,N),max_inds] = True

	pts_enc_1hot_nd = rz.unflatten_2d_array(pts_enc_1hot_flt,pts_enc_nd,axis=axis)

	return pts_enc_1hot_nd

	# def decode_argmax_points_mtx_nd(self,pts_enc_nd,axis=1):
		# shp = pts_enc_nd
		# if()
		# np.argmax(pts_enc_nd)

# ************************************
# ***** Invert LAB colors to RGB *****
# ************************************
def highlight_lab_grid(l_vals,ab_grid=-1,returnGrid=False,minPercentile=25,useLog=True):
	# min_val = np.min(l_vals.flatten()[l_vals.flatten()!=0])
	min_val = np.percentile(l_vals.flatten()[l_vals.flatten()!=0],minPercentile)
	# print min_val
	if(useLog):
		l_grid = np.log10(l_vals+min_val)
	else:
		l_grid = l_vals
	# l_grid = np.sqrt(l_grid)
	l_grid = 20+60.*(l_grid-np.min(l_grid))/(np.max(l_grid)-np.min(l_grid))
	# return l_grid
	return show_lab_grid(l_grid,ab_grid,returnGrid=returnGrid)

def show_lab_grid(l_grid,grid=-1,returnGrid=False):
	if(rz.check_value(grid,-1)):
		grid = rz.grid_ab(2)

	# l_grid = l+np.zeros_like(grid.ab_grid)[:,:,[0]]
	ab_grid = grid.ab_grid
	# print np.array(l_grid).ndim
	if(np.array(l_grid).ndim==0):
		l_grid = l_grid+np.zeros_like(ab_grid)
		l_grid = l_grid[:,:,[0]]
		# print l_grid.shape
	if(l_grid.ndim==2):
		l_grid = l_grid[:,:,rz.na()]
	A = ab_grid.shape[0]
	B = ab_grid.shape[1]
	# l_grid = caffe.io.resize(l_grid,(A,B,1))

	lab_grid = np.concatenate((l_grid,ab_grid),axis=2).astype('float64')
	rgb_grid = (255*np.maximum(np.minimum(color.lab2rgb(lab_grid),1),0)).astype('uint8')

	if(returnGrid):
		return (rgb_grid,grid)
		# return (lab_grid,grid)
	else:
		return rgb_grid

def show_ab_grid(l,grid=-1,returnGrid=False,isGridObj=True):
	if(rz.check_value(grid,-1)):
		grid = rz.grid_ab(2)

	if(isGridObj):
		ab_grid = grid.ab_grid
	else:
		ab_grid = grid
	l_grid = l+np.zeros_like(ab_grid)[:,:,[0]]

	lab_grid = np.concatenate((l_grid,ab_grid),axis=2).astype('float64')
	rgb_grid = (255*np.maximum(np.minimum(color.lab2rgb(lab_grid),1),0)).astype('uint8')

	if(returnGrid):
		return (rgb_grid,grid)
		# return (lab_grid,grid)
	else:
		return rgb_grid

def ab2sqrtpolar(pts_flt):
	pts_flt_polar_sqrt = np.zeros_like(pts_flt)
	pts_flt_polar_sqrt[:,0] = np.sqrt(np.sqrt(np.sum(pts_flt**2,axis=1)))
	pts_flt_polar_sqrt[:,1] = np.arctan2(pts_flt[:,0],pts_flt[:,1])*180/np.pi
	return pts_flt_polar_sqrt

def sqrtpolar2ab(pts_flt_polar):
	pts_flt_polar = pts_flt_polar.copy()
	pts_flt_polar[:,0] = pts_flt_polar[:,0]**2
	pts_flt_polar[:,1] = pts_flt_polar[:,1]*np.pi/180

	pts_flt = np.zeros_like(pts_flt_polar)
	pts_flt[:,0] = pts_flt_polar[:,0]*np.sin(pts_flt_polar[:,1])
	pts_flt[:,1] = pts_flt_polar[:,0]*np.cos(pts_flt_polar[:,1])

	return pts_flt

	# pts_nd = rz.unflatten_2d_array(pts_flt,pts_enc_nd,axis=axis)

def set_us_layer(net,us_layers):
	us_kern = np.array([[0.25,0.5,0.25],[0.5,1.,0.5],[0.25,0.5,0.25]])
	# us_kern = us_kern/np.sum(us_kern)
	for layer in us_layers.flatten():
		net.params[layer][0].data[...] = 0
		C = net.params[layer][0].data.shape[0]
		net.params[layer][1].data[...] = 0
		for cc in range(C):
			net.params[layer][0].data[cc,cc,:3,:3] = us_kern

