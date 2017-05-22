#!/usr/bin/env python
# Richard Zhang / 2015.08.18
# Utility functions

import numpy as np
import os
import scipy
import itertools
import scipy.misc
import time
import datetime
import caffe
import matplotlib.pyplot as plt
import csv
from skimage import color

# Class converts images in XxYxCxN, RGB format to caffe input format
# Caffe format is NxCxXxY, BGR, mean centered
# Note that this I had an error in my old version from normal convention (switched X and Y)
class img_data_transformer_corr():
	def __init__(self, use_imgnet_mean=True, verbose=False):
		self.use_imgnet_mean = use_imgnet_mean
		if(use_imgnet_mean):
			if(verbose):
				print("Using Imagenet convention")
			# dummyMeanFile = '/home/eecs/rich.zhang/src/libs/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
			dummyMeanFile = '/home/eecs/rich.zhang/src/libs/caffe5/python/caffe/imagenet/ilsvrc_2012_mean.npy'
			self.MEAN = np.load(dummyMeanFile).mean(1).mean(1)
		else:
			if(verbose):
				print("Using Places convention")
			dummyMeanFile = '/home/eecs/rich.zhang/data_rzhang/models/caffe/placesCNN_upgraded/places_mean.mat'
			self.MEAN = scipy.io.loadmat(dummyMeanFile)['image_mean'].mean(0).mean(0)

	def img2data(self,img):
		if(img.ndim==3):
			toExp = True
			img = img[:,:,:,na()]
		else:
			toExp = False

		toRet = img[:,:,[2,1,0],:].transpose((3,2,0,1)) - self.MEAN[np.newaxis,:,np.newaxis,np.newaxis]

		if(toExp):
			return toRet[0,:,:,:]
		else:
			return toRet

	def data2img(self,data):
		if(data.ndim==3):
			toExp = True
			data = data[na(),:,:,:]
		else:
			toExp = False

		# return data[:,[2,1,0],:,:].transpose((3,2,1,0)) + self.MEAN[np.newaxis,[2,1,0],np.newaxis,np.newaxis]
		toRet = data[:,[2,1,0],:,:].transpose((2,3,1,0)) + self.MEAN[np.newaxis,np.newaxis,[2,1,0],np.newaxis]

		if(toExp):
			return np.clip(toRet[:,:,:,0],0,255).astype('uint8')
		else:
			return np.clip(toRet.astype('uint8'),0,255)

# Class converts images in XxYxCxN, RGB format to caffe input format
# Caffe format is NxCxYxX, BGR format, mean centered
class img_data_transformer():
	def __init__(self, use_imgnet_mean=True, verbose=False):
		self.use_imgnet_mean = use_imgnet_mean
		if(use_imgnet_mean):
			if(verbose):
				print("Using Imagenet convention")
			# dummyMeanFile = '/home/eecs/rich.zhang/src/libs/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
			dummyMeanFile = '/home/eecs/rich.zhang/src/libs/caffe5/python/caffe/imagenet/ilsvrc_2012_mean.npy'
			self.MEAN = np.load(dummyMeanFile).mean(1).mean(1)
		else:
			if(verbose):
				print("Using Places convention")
			dummyMeanFile = '/home/eecs/rich.zhang/data_rzhang/models/caffe/placesCNN_upgraded/places_mean.mat'
			self.MEAN = scipy.io.loadmat(dummyMeanFile)['image_mean'].mean(0).mean(0)

	def img2data(self,img):
		if(img.ndim==3):
			toExp = True
			img = img[:,:,:,na()]
		else:
			toExp = False

		if(self.use_imgnet_mean):
			toRet = img[:,:,[2,1,0],:].transpose((3,2,1,0)) - self.MEAN[np.newaxis,:,np.newaxis,np.newaxis]
		else: # places
			toRet = img[:,:,[2,1,0],:].transpose((3,2,0,1)) - self.MEAN[np.newaxis,:,np.newaxis,np.newaxis]

		if(toExp):
			return toRet[0,:,:,:]
		else:
			return toRet

	def data2img(self,data):
		if(data.ndim==3):
			toExp = True
			data = data[na(),:,:,:]
		else:
			toExp = False

		# return data[:,[2,1,0],:,:].transpose((3,2,1,0)) + self.MEAN[np.newaxis,[2,1,0],np.newaxis,np.newaxis]
		if(self.use_imgnet_mean):
			toRet = data[:,[2,1,0],:,:].transpose((3,2,1,0)) + self.MEAN[np.newaxis,np.newaxis,[2,1,0],np.newaxis]
		else:
			toRet = data[:,[2,1,0],:,:].transpose((2,3,1,0)) + self.MEAN[np.newaxis,np.newaxis,[2,1,0],np.newaxis]

		if(toExp):
			return toRet[:,:,:,0].astype('uint8')
		else:
			return toRet.astype('uint8')

class img_data_lab_transformer():
	def __init__(self, verbose=False, corr=True):
		self.MEAN = np.array((50,0,0))
		self.corr = corr # correct ordering of spatial axes
		if(verbose):
			print('Converting between rgb to lab data format')

	def img2data(self,imgs):
		if(imgs.ndim==3):
			toExp = True
			imgs = imgs[:,:,:,na()]
		else:
			toExp = False

		imgs_lab = np.zeros_like(imgs,dtype='float64')
		# leta = 
		for ii in range(imgs.shape[3]):
			imgs_lab[:,:,:,ii] = color.rgb2lab(imgs[:,:,:,ii])

		imgs_lab = imgs_lab - self.MEAN[na(),na(),:,na()]
		if(self.corr):
			data = imgs_lab.transpose((3,2,0,1))
		else:
			data = imgs_lab.transpose((3,2,1,0))

		if(toExp):
			return data[0,:,:,:]
		else:
			return data

	def data2img(self,data):
		if(data.ndim==3):
			toExp = True
			data = data[na(),:,:,:]
		else:
			toExp = False

		if(self.corr):
			imgs_lab = np.float64(data.transpose((2,3,1,0)) + self.MEAN[na(),na(),:,na()])
		else:
			imgs_lab = np.float64(data.transpose((3,2,1,0)) + self.MEAN[na(),na(),:,na()])
		imgs = np.zeros_like(imgs_lab,dtype='float64')
		for ii in range(imgs_lab.shape[3]):
			imgs[:,:,:,ii] = color.lab2rgb(imgs_lab[:,:,:,ii])
		imgs = np.minimum(np.maximum(imgs,0),1)
		imgs = np.uint8(255*imgs)

		if(toExp):
			return imgs[:,:,:,0]
		else:
			return imgs

	def wtf():
		print(0)

# Timing class for tic toc operation
class Timer():
	def __init__(self):
		self.cur_t = time.time()

	def tic(self):
		self.cur_t = time.time()

	def toc(self):
		return time.time()-self.cur_t

	def tocStr(self, t=-1):
		if(t==-1):
			return str(datetime.timedelta(seconds=np.round(time.time()-self.cur_t,3)))[:-4]
		else:
			return str(datetime.timedelta(seconds=np.round(t,3)))[:-4]

class BatchNorm1D():
	def __init__(self,vals,toCalcParams=True):
		if(toCalcParams):
			self.mean = np.mean(np.float32(vals))
			self.std = np.std(np.float32(vals))
		else:
			self.mean = vals[0]
			self.std = vals[1]

	def normalize(self,observations):
		return (observations-self.mean)/self.std

class ObservationNorms():
	def __init__(self, vals, mean=' ', std=' '):
		if(check_value(mean,' ') and check_value(std, ' ')):
			self.bn = BatchNorm1D(vals)
		else:
			self.bn = BatchNorm1D([mean,std],toCalcParams=False)
		self.vals = vals
		self.vals_norm = self.bn.normalize(vals)

# Timing class for estimating time remaining
class LoopETA():
	def __init__(self, N, B):
		# INPUTS
		# 	N 		scalar  	total number of iterations
		# 	B 		scalar 		number of iterations to update display timing output
		self.t = Timer()
		self.N = N
		self.B = B

	def print_update(self, n, pre=""):
		# INPUTS
		# 	n 		scalar 		current iteration number
		# OUTPUTS - console
		# 	print string
		if(pre!=""):
			pre+=" "
		if(np.mod(n,self.B)==0 or n==self.N-1): # every B iterations or in the last iteration
			print(pre+str(n)+"/"+str(self.N)+": "+self.t.tocStr()+"/"+self.t.tocStr(t=1.0*(self.N/(n+1.)*self.t.toc())))
			return True
		return False

# Indent class keeps track of indentions, for outputting text
class Indent():
	def __init__(self,N=2):
		self.n = 0
		self.N = N
		self.ind = ' '*N

	def increment(self, n=1):
		self.n+=n
		self.ind+=' '*self.N

	def decrement(self, n=1):
		self.n = np.maximum(0, self.n-n)
		self.ind = ' '*self.N*self.n

	def get_ind(self):
		return self.ind

	def set_ind(self, N):
		self.N = N
		self.ind = ' '*N*self.n

# Batcher class calculates indices when processing mini-batches
# PUBLIC VARIABLES
# 	N 			scalar 			number of elements total
# 	batchsize 	scalar 			number of elements maximum per batch
# 	ends 		Bx2 			upper and lower index bounds each minibatch, first number inclusive, second exclusive
# 	c 			scalar 			current batch number
# 	cur_ends 	2 vector		end indices of current batch
# 	cur_I 		scalar			current number of indices
# 	cur_inds 	cur_I vector 	current number of inidces
# EXAMPLE USE
# 	batch = Batcher(1000, 50) # initiate batcher
# 	for ii in range(batch.B):
# 		batch.increment()
# 		# process(array_to_process[batch.inds])
class Batcher():
	# INPUTS - init
	# 	N 					scalar 		number of elements total
	# 	batchsize 			scalar 		number of elements maximum per batch
	# 	update_interval 	scalar 		[1] print update interval
	def __init__(self, N, batchsize, update_interval=1, update_text=''):
		self.N = np.uint64(N)
		self.batchsize = np.uint64(batchsize)
		self.ends = np.uint64(np.r_[(np.arange(0,N,self.batchsize),N)])
		self.B = self.ends.shape[0]-1 # number of batches
		self.c = -1
		self.leta = LoopETA(self.B, update_interval)
		self.update_text = update_text

	# INPUTS - increment
	# 	c 			scalar 		batch number. if specified, internal variables will reflect this minibatch
	# 							if unspecified, minibatch will automatically increment to next one. will wrap around
	def increment(self, c=-1, TO_PRINT=False):
		if(c!=-1):
			self.c = c
		else:
			self.c+=1

		if(self.c==self.B):
			self.c=0

		self.cur_ends = self.ends[self.c:self.c+2] # upper and lower bounds of current indices
		self.cur_inds = np.uint64(np.arange(self.cur_ends[0],self.cur_ends[1])) # current indices
		self.cur_I = self.cur_ends[1]-self.cur_ends[0] # current number of elements
		self.cur_mask = np.zeros(self.batchsize, dtype='bool')
		self.cur_mask[0:self.cur_I] = True	

		if(TO_PRINT):
			self.print_update()

		return self.cur_inds

	def increment_full(self, c=-1, TO_PRINT=False):
		# return a set of indices which are a "full" batchsize
		self.increment(c, TO_PRINT) # increment

		if(self.cur_I!=self.batchsize): # augment if needed
			return np.repeat(self.cur_inds, np.ceil(1.0*self.batchsize/self.cur_I))[0:self.batchsize]
		return self.cur_inds

	def inds_str(self):
		return '%s_%s'%(prettify_num(self.cur_inds[0],forFile=True),prettify_num(self.cur_inds[-1],forFile=True))

	def print_update(self):
		return self.leta.print_update(self.c,pre=self.update_text)

# Class selects indides for cross-validation
class CrossvalIndices():
	def __init__(self, N, K, INCL_TST=False):
		self.N = N
		self.K = K
		self.k_inds = np.random.randint(K, size=N) # randomize indices
		self.INCL_TST = INCL_TST

	def trn_mask(self,k):
		if(self.INCL_TST):
			return (~self.val_mask(k)) * (~self.tst_mask(k))
		else:
			return self.k_inds!=k

	def val_mask(self,k):
		if(self.INCL_TST):
			return self.k_inds==np.mod(k-1, self.K)
		else:
			return self.k_inds==k

	def tst_mask(self,k):
		if(self.INCL_TST):
			return self.k_inds==k
		else:
			return np.zeros_like(self.k_inds,dtype='bool')

	def save(self, filepath):
		np.save(filepath, self.k_inds)

	def load(self, filepath):
		self.k_inds = np.load(filepath, self)


# class RandSplitIndices():
	# def __init__(self, N, K, INCL_TST=False):

def na(): # shorthand for new axis
	return np.newaxis

# ***********************************************
# *************** IMAGE FUNCTIONS ***************
# ***********************************************
def montage(imgs,PAD=5,RATIO=16/9.,EXTRA_PAD=(False,False),MM=-1,NN=-1,primeDir=0,verbose=False,returnGridPos=False,backClr=np.array((0,0,0))):
	# INPUTS
	# 	imgs 		YxXxMxN or YxXxN 	
	# 	PAD 		scalar				number of pixels in between
	# 	RATIO 		scalar 				target ratio of cols/rows
	# 	MM 			scalar 				# rows, if specified, overrides RATIO
	# 	NN 			scalar 				# columns, if specified, overrides RATIO
	# 	primeDir 	scalar 				0 for top-to-bottom, 1 for left-to-right
	# OUTPUTS
	# 	mont_imgs 	MM*Y x NN*X x M 	big image with everything montaged
	# def montage(imgs, PAD=5, RATIO=16/9., MM=-1, NN=-1, primeDir=0, verbose=False, forceFloat=False):
	if(imgs.ndim==3):
		toExp = True
		imgs = imgs[:,:,np.newaxis,:]
	else:
		toExp = False

	Y = imgs.shape[0]
	X = imgs.shape[1]
	M = imgs.shape[2]
	N = imgs.shape[3]

	PADS = np.array((PAD))
	if(PADS.flatten().size==1):
		PADY = PADS
		PADX = PADS
	else:
		PADY = PADS[0]
		PADX = PADS[1]

	if(MM==-1 and NN==-1):
		NN = np.ceil(np.sqrt(1.0*N*RATIO))
		MM = np.ceil(1.0*N/NN)
		NN = np.ceil(1.0*N/MM)
	elif(MM==-1):
		MM = np.ceil(1.0*N/NN)
	elif(NN==-1):
		NN = np.ceil(1.0*N/MM)

	if(verbose):
		print(str(MM)+" "+str(NN))

	if(primeDir==0): # write top-to-bottom
		[grid_mm, grid_nn] = np.meshgrid(np.arange(MM,dtype='uint'), np.arange(NN,dtype='uint'))
	elif(primeDir==1): # write left-to-right
		[grid_nn, grid_mm] = np.meshgrid(np.arange(NN,dtype='uint'), np.arange(MM,dtype='uint'))

	grid_mm = np.uint(grid_mm.flatten()[0:N])
	grid_nn = np.uint(grid_nn.flatten()[0:N])

	EXTRA_PADY = EXTRA_PAD[0]*PADY
	EXTRA_PADX = EXTRA_PAD[0]*PADX

	# mont_imgs = np.zeros(((Y+PAD)*MM-PAD, (X+PAD)*NN-PAD, M), dtype=use_dtype)
	mont_imgs = np.zeros((np.uint((Y+PADY)*MM-PADY+EXTRA_PADY),np.uint((X+PADX)*NN-PADX+EXTRA_PADX),M), dtype=imgs.dtype)
	mont_imgs = mont_imgs+backClr.flatten()[na(),na(),:].astype(mont_imgs.dtype)

	for ii in np.random.permutation(N):
		# print imgs[:,:,:,ii].shape
		# mont_imgs[grid_mm[ii]*(Y+PAD):(grid_mm[ii]*(Y+PAD)+Y), grid_nn[ii]*(X+PAD):(grid_nn[ii]*(X+PAD)+X),:]
		mont_imgs[np.uint(grid_mm[ii]*(Y+PADY)):np.uint((grid_mm[ii]*(Y+PADY)+Y)), np.uint(grid_nn[ii]*(X+PADX)):np.uint((grid_nn[ii]*(X+PADX)+X)),:] = imgs[:,:,:,ii]

	if(M==1):
		imgs = imgs.reshape(imgs.shape[0], imgs.shape[1], imgs.shape[3])

	if(toExp):
		mont_imgs = mont_imgs[:,:,0]

	if(returnGridPos):
		# return (mont_imgs,np.concatenate((grid_mm[:,:,np.newaxis]*(Y+PAD), grid_nn[:,:,np.newaxis]*(X+PAD)),axis=2))
		return (mont_imgs, np.concatenate((grid_mm[:,np.newaxis]*(Y+PADY), grid_nn[:,np.newaxis]*(X+PADX)),axis=1))
		# return (mont_imgs, (grid_mm,grid_nn))
	else:
		return mont_imgs

def mont_2d(imgs,M=2,MM=-1,NN=-1,PAD=5,primeDir=0,returnGridPos=False):
	# Extention of montage function above.
	# Assume input images were originally in a grid with 2nd dimension M. Reshape and concatenate
	# so that matching images are above each other. Then run montage function as usual.
	# INPUTS
	# 	imgs 	YxXx3xNM
	#	M 		scalar 	
	# 	NN 		way to arrange the montage
	(Y,X,C,NM) = imgs.shape
	N = NM/M
	imgs = np.concatenate(imgs.reshape((Y,X,3,N,M)).transpose((4,0,1,2,3)), axis=0)
	return montage(imgs,MM=MM,NN=NN,PAD=PAD,primeDir=primeDir,returnGridPos=returnGridPos)

def grid_text(grid_pos,strs,values,colors=-1,OFF=20,FONTSIZE=6):
	# INPUTS
	# 	grid_pos 	Nx2 		position to write in pixels
	# 	strs 		S array		number of string templates to write out
	# 	values		S tuple of N array 	<this has to be a tuple!!>
	# 	OFF 		scalar 		offset to apply
	# 	FONTSIZE 	scalar 		fontsize to use
	# 	<...add other stuff in the future to make more versatile...>
	strs = np.array(strs)
	if(check_value(colors,-1)):
		colors = ['r']*len(strs)
	N = grid_pos.shape[0]
	S = strs.size
	for nn in range(N):
		for s in range(S):
			plt.text(grid_pos[nn,1], grid_pos[nn,0]+OFF*s, strs.flatten()[s]%values[s][nn],
				verticalalignment='top', horizontalalignment='left', fontweight='bold', fontsize=FONTSIZE, color=colors[s])


def fill_border(imgs, clr, BORDER=10):
	clr = np.array(clr)
	for cc in range(3):
		imgs[:BORDER,:,cc,:] = clr[cc]
		imgs[-BORDER:,:,cc,:] = clr[cc]
		imgs[:,:BORDER,cc,:] = clr[cc]
		imgs[:,-BORDER:,cc,:] = clr[cc]
	return imgs

def big_splat_img(imgs, xys, xs=-1, ys=-1, scale=-1, scaleFactor=4., snap=False, snapAmt=-1, border=0, borderClr=(0,0,0), returnGridPos=False):
	# INPUTS
	# 	imgs 		YxXx3xN 	images
	# 	xys 		Nx2 		locations to splat
	# 	xs 	 		tuple 		x bounds
	# 	ys 			tuple 		y bounds
	# 	scale 		scalar 		pixels per unit in locations
	# 	snap 		boolean 	whether to snap or not
	# OUTPUTS
	# 	big_img 	
	imgs = imgs.copy()
	xys = xys.copy()
	(Y,X,M,N) = imgs.shape
	# imgs = fill_border(imgs, clr=borderClr, BORDER=border)

	if(check_value(xs,-1)):
		xs = np.max(np.abs(xys[:,0]))
	if(check_value(ys,-1)):
		ys = np.max(np.abs(xys[:,1]))

	if(np.size(xs)==1):
		xs = (-xs,xs)
	if(np.size(ys)==1):
		ys = (-ys,ys)

	if(scale==-1):
		# scale = np.max((Y,X)) # amount to scale the space by
		scale = scaleFactor*np.sqrt((N*Y*X)/((ys[1]-ys[0])*(xs[1]-xs[0])))

	if(snap==True): # snap to a grid
		if(snapAmt==-1): # determine the pixel coarseness to snap to
			snapAmt = 1.0*np.array((X,Y))/scale
			# print snapAmt
		xys = np.round(1.0*xys/snapAmt[np.newaxis,:])*snapAmt[np.newaxis,:]

	big_img = np.zeros(((ys[1]-ys[0])*scale+Y, (xs[1]-xs[0])*scale+X, 3), imgs.dtype)

	gridPos = np.zeros((N,2),dtype='uint')
	leta = LoopETA(N,100)
	for nn in range(N):
		x = xys[nn,0]
		y = xys[nn,1]
		if(x>=xs[0] and x<xs[1] and y>=ys[0] and y<ys[1]): # include image if it's inside the bounds
			y_off = np.round(1.0*(y-ys[0])*scale).astype('int')
			x_off = np.round(1.0*(x-xs[0])*scale).astype('int')

			# print (x, y, xs[0], ys[0], x_off, y_off, Y, X)
			# print big_img.shape
			# print big_img[y_off:y_off+Y, x_off:x_off+X,:].shape
			# print np.max(imgs[:,:,:,nn])

			big_img[y_off:y_off+Y, x_off:x_off+X,:] = imgs[:,:,:,nn]
			gridPos[nn,:] = np.array(y_off,x_off)

		leta.print_update(nn,pre='Splatting...')

	if(returnGridPos):
		return (big_img, gridPos)
	else:
		return big_img

def load_resize_img(filepath, imsize=-1):
	# INPUTS
	# 	filepath 	string 		filepath
	# 	imsize 		scalar 		[-1] size to resize image to (Y,X), -1 for no resizing
	imsize = np.array(imsize)
	img = 255*caffe.io.load_image(filepath)
	(Y,X) = img[:,:,0].shape
	if(check_value(imsize,-1)):
		return img
	elif(imsize.ndim==1):
		imsize = np.zeros(2)+imsize		
	return scipy.ndimage.interpolation.zoom(img, (1.*imsize[0]/Y, 1.*imsize[1]/X, 1), order=0, mode='constant', cval=0.0, prefilter=False)

def prettify_nums(numbers, numDec=3, forFile=False, prefix='', suffix=''):
	# strs = np.zeros_like(numbers, dtype='s'+str(10+len(prefix)+len(suffix)))
	strs = np.zeros(numbers.size, dtype='|S'+str(10+len(prefix)+len(suffix)))
	for (nn,number) in enumerate(numbers):
		strs[nn] = prettify_num(number, numDec, forFile, prefix, suffix)
	return strs

def prettify_num(number,numDec=3,forFile=False, prefix='', suffix=''):
	if(number==0):
		return prefix+"0"+suffix

	if(number<0):
		if(forFile):
			negStr = 'n'
		else:
			negStr = '-'
		number = -number
	else:
		negStr = ''
	# if(not check_value(log10_val,np.Infinity)):
	log10_val = np.round(1.*np.log10(number))
	suffixes = ('f','p','n','u','m','','k','M','G','T')
	suff_vals = np.array((-15.,-12.,-9.,-6.,-3.,0.,3.,6.,9.,12.))
	
	log10_ind = find_nearest_ind(suff_vals, log10_val)

	numPort = str(np.round(1.*number/10**suff_vals[log10_ind],numDec)).replace('[','').replace(']','').replace(' ','')
	# numPort = '%.'+str(numDec)+'f'%(number/10**suff_vals[log10_ind])
	# numPort = numPort.replace('[','').replace(']','').replace(' ','')
	if(numPort[-1]=='.'):
		numPort = numPort[:-1]
	if(forFile):
		numPort = numPort.replace('.','p')
	suffPort = suffixes[log10_ind[0]]
	return prefix+negStr+numPort+suffPort+suffix

def remove_border(img_path):
	# Remove white image border from a saved file
	# When figures are saved, there is an annoying white border.
	img = caffe.io.load_image(img_path)
	img_white = np.sum(img==1,axis=2)==3
	(Y,X) = img_white.shape
	x_ends = np.where(np.sum(img_white,axis=0)!=Y)[0][[1,-1]]
	y_ends = np.where(np.sum(img_white,axis=1)!=X)[0][[1,-1]]
	plt.imsave(img_path, img[y_ends[0]:y_ends[1],x_ends[0]:x_ends[1],:])

def find_nearest_ind(ref_array, srch_array, returnDeltas=False):
	# Search for nearest index
	# INPUTS
	# 	ref_array 			N array 		reference array to search in
	# 	srch_array 			S array 		values to search for
	# 	returnDeltas 		bool 			[False] whether to return deltas
	# OUTPUTS - returned
	# 	[0] 				S array 		indices in reference array which closest match srch_array
	# 	[1] 				S array 		returned if returnDeltas is true
	srch_array = np.array(srch_array).flatten()
	srch_inds = np.zeros(srch_array.size, dtype=np.uint)
	for ss in range(srch_array.size):
		srch_inds[ss] = np.argmin(np.abs(ref_array-srch_array[ss]))

	if(returnDeltas):
		deltas = ref_array[srch_inds]-srch_array
	
	if(returnDeltas):
		return (srch_inds, deltas)
	else:
		return srch_inds

def moving_avg(vals, N):
	# INPUTS
	#  	vals  	1d array
	# 	N 		integer
	mask = ~np.isinf(vals)
	use_vals = vals.copy()
	use_vals[~mask] = 0
	ones = 1.*np.ones(N)
	tmp = np.convolve(use_vals, 1.*ones, 'same') / np.convolve(mask, ones, 'same')
	return tmp[:vals.size]

def set_diagonal(in_mtx,value):
	in_mtx[np.arange(0,in_mtx.shape[0]),np.arange(0,in_mtx.shape[1])] = value
	return in_mtx

def min_max(vals,axis=0):
	return np.array(np.min(vals,axis=axis),np.max(vals,axis=axis))

def kmutual_nn(dist_mtx,K=1,inds0=-1,inds1=-1,isMutual=True,isSymm=False,returnRawInds=False):
	# Find mutual NNs are within each others top K
	# INPUTS
	# 	dist_mtx 	M0xM1 array 		distance matrix
	# 	K 			scalar 				[1] K-NN
	#	inds0 		M0 array 			original indices, -1 for 0:M0
	# 	inds1 		M1 array 			original indices, -1 for 0:M1
	# 	isMutual 	bool 				[True] requires indices be each others NNs
	# 	isSymm		bool				[False] indicates that matrix is symmetric, and will only return each pair once
	# OUTPUTS - returned
	# 	[0] 		Nx2 				indices which are mutual NNs
	# 	[1] 		N 					distance of matches
	(M0,M1) = dist_mtx.shape
	if(check_value(inds0,-1)):
		inds0 = np.arange(0,M0)
	if(check_value(inds1,-1)):
		inds1 = np.arange(0,M1)

	if(isSymm):
		dist_mtx[np.arange(0,M0),np.arange(0,M1)] = np.Infinity # infinity out diagonal

	# for each set, find nns in other set
	pool0_closest_ind1s = np.argsort(dist_mtx,axis=1) # each row is the indices ordered by NNs
	pool1_closest_ind0s = np.argsort(dist_mtx,axis=0) # each column are indices ordered by NNs

	is_nn0 = np.zeros((M0,M1),dtype='bool')
	is_nn1 = np.zeros((M0,M1),dtype='bool')
	is_nn0[np.arange(0,M0)[:,np.newaxis],pool0_closest_ind1s[:,0:K]] = True
	is_nn1[pool1_closest_ind0s[0:K,:], np.arange(0,M1)[np.newaxis,:]] = True
	if(isMutual):
		mutual_nn_inds = find_nd(is_nn0*is_nn1)
	else:
		mutual_nn_inds = find_nd(is_nn0+is_nn1)

	dist_nns = dist_mtx[mutual_nn_inds[:,0],mutual_nn_inds[:,1]]

	dist_sortinds = np.argsort(dist_nns)
	dist_matches = dist_nns[dist_sortinds]
	mutual_nn_inds = mutual_nn_inds[dist_sortinds,:]

	if(isSymm): # filter out redundancies
		mutual_nn_inds = np.sort(mutual_nn_inds) # sort them first
		mutual_nn_lin_inds = sub2ind2(mutual_nn_inds,(M0,M1))
		un_inds = np.unique(mutual_nn_lin_inds, return_index=True)[1]
		un_inds = np.sort(un_inds)

		mutual_nn_inds = mutual_nn_inds[un_inds,:]
		dist_matches = dist_matches[un_inds]

	mutual_nn_inds = mutual_nn_inds[dist_matches!=np.Infinity]
	dist_matches = dist_matches	[dist_matches!=np.Infinity]

	# index into larger index group
	pool0_match_inds = inds0[mutual_nn_inds[:,0]]
	pool1_match_inds = inds1[mutual_nn_inds[:,1]]

	retList = [np.c_[[pool0_match_inds,pool1_match_inds]].transpose(1,0), dist_matches]
	if(returnRawInds):
		retList.append(mutual_nn_inds)
	return retList

# *****************************
# ***** CALCULATE L2 NORM *****
# *****************************
def calc_norm(fts, verbose=False):
	# Calculates norm of an N-d feature vector. Batched so that it works on HDF5 datasets too
	# 	** DOESNT CURRENTLY WORK ON N-D VECTOR, ONLY 2D **
	# INPUTS
	# 	fts 		Nd array, Nxwhatever
	# OUTPUTS - returned
	# 	[0] 		N array 		norm values
	(N,F) = fts.shape
	norms = np.zeros(N, dtype='float32')
	batchsize = np.uint64(2e6/F)
	b = Batcher(N,batchsize,update_interval=100)
	for bb in range(b.B):
		b_inds = b.increment(TO_PRINT=verbose)
		norms[b_inds] = np.sqrt(np.sum(np.float64(fts[b_inds,:])**2,axis=1))
	return norms

def batch_L2_distance(fts0, fts1, verbose=True):
	batchsize = 100
	(N0,F) = fts0.shape
	(N1,F) = fts1.shape

	dists = np.zeros((N0,N1), dtype='float32')
	leta = rz.LoopETA(N0,25)
	for nn in range(N0):
		dists[nn,:] = np.linalg.norm(fts0[nn,:][np.newaxis,:] - fts1)
		leta.print_update(nn)
	return dists

def normalize(in_array, axes=1, eps=0):
	# L2 normalize an input vector - CURRENTLY BROKEN FOR MULTIPLE AXES =()
	# INPUTS
	# 	in_array 	nd-array
	# 	axes  		vector 		dimensions to normalize away
	# OUTPUTS
	# 	out_array 	nd-array
	axes = np.array(axes).flatten()
	axes = axes[axes<len(in_array.shape)]

	out_array = in_array[...]
	for ii in range(axes.size):
		# out_array = out_array / (np.expand_dims(np.linalg.norm(out_array,axis=axes[ii]),axes[ii]) + eps)
		out_array = out_array / (np.expand_dims(np.sqrt(np.sum(out_array**2,axis=axes[ii])),axes[ii]) + eps)
		# print out_array
	return out_array

	# # sanity check code
	# in_array = np.random.randn(5,3)
	# new_array = normalize(in_array, axes=np.array((1)))
	# np.sum(new_array**2,axis=1)

	# in_array = np.ones((1,3,2,4))
	# new_array = normalize(in_array, axes=np.array((1,2,3)))
	# np.sum(np.sum(np.sum(new_array**2,axis=1),axis=1),axis=1)

	# axis = 1 # direction of vectors
	# np.linalg.norm(in_array, axis=axis)
	# np.expand_dims(np.linalg.norm(in_array, axis=axis),axis)
	# eps = 0
	# new_array = in_array / (np.expand_dims(np.linalg.norm(in_array,axis=axis),axis) + eps)
	# np.linalg.norm(new_array, axis=axis)

# **********************************************
# *************** PATH FUNCTIONS ***************
# **********************************************
def mkdir(path):
	# make directory
	if not os.path.exists(path):
		os.makedirs(path)

def listdir(path, ext=''):
	# list elements directory
	# INPUTS
	# 	path
	# 	ext 		extension
	# OUTPUTS
	# 	string array of files

	files = np.sort(np.array(os.listdir(path)))
	if(ext==''):
		return files
	else:
		if(ext[0]=='.'):
			return files[mask_ext(files, ext)]
		else:
			return files[mask_ext(files, '.'+ext)]

def mask_ext(files, ext):
	# mask files which have a certain extension
	return np.array([os.path.splitext(file) for file in files])[:,1]==ext

# def moving_avg(vals, N):
# 	# Moving average which takes care of edge effects
# 	temp = np.ones(vals.size)
# 	temp[0:N/2] = 1.0*np.arange(N/2,N)/N
# 	temp[-N/2:] = 1.0*np.arange(N,N/2,-1)/N
# 	return np.convolve(vals, 1.0*np.ones(N)/N, 'same') / temp

def list_permutations(N):
	# OUTPUT
	# 	N! x N matrix
	perm_list = np.zeros((scipy.misc.factorial(N), N),dtype='uint')

	cnt = -1
	for p in itertools.permutations(np.arange(0,N)):
		cnt+=1
		perm_list[cnt,:] = p
	return perm_list

def ind2sub2(inds, dims):
	# INPUTS
	# 	inds 	N array
	# 	dims  	2 tuple or array
	# OUTPUT - returned
	# 	[ ] 	Nx2 array
	shp = inds.shape
	inds = inds.flatten()
	temp = np.unravel_index(inds, dims=dims)
	# val = np.c_[[temp[0],temp[1]]]
	# for a in np.arange(2,len(dims)):
		# val = np.c_[[val,temp[a]]]
	val = np.zeros((temp[0].size,len(temp)))
	for a in range(len(temp)):
		val[:,a] = temp[a]

	new_shp = np.concatenate((np.array(shp),np.array(len(temp)).flatten()),axis=0)
	return val.reshape(new_shp).astype('int')
	# return val.transpose(1,0)
	# return np.c_[[temp[0],temp[1]]].transpose(1,0)

def sub2ind2(subs, dims):
	# INPUTS
	# 	subs 	Nx2 array
	# 	dims  	2 tuple or array
	# OUTPUT - returned
	# 	[ ] 	N array
	if(subs.ndim==1):
		subs = subs[np.newaxis,:]
	tmp = np.zeros_like(subs,dtype='int')
	tmp[...] = subs[...]
	subs = tmp
	# return np.ravel_multi_index((subs[:,0],subs[:,1]), dims)
	
	inds = np.ravel_multi_index((subs[:,0],subs[:,1]), dims)
	return inds

def str2ind(in_str):
	# INPUTS
	# 	in_str 		N strings
	# OUTPUTS - returned
	# 	[0] 		N ints 				indexing into second output
	# 	[1]			S uinque strings
	# in_str = out[1][out[0]]
	strs = np.unique(in_str)
	S = strs.size # number of unique strings
	N = in_str.size # number of strings

	cat_out = np.zeros(N,dtype='uint64')
	for ss in range(S):
		cat_out[in_str==strs[ss]] = ss

	return (cat_out,strs)

def calc_percentile(in_var,per):
	sort_var = np.sort(in_var.flatten())
	N = sort_var.size
	ind = np.minimum(np.maximum(0,np.int(N*per)),N-1)
	return sort_var[np.int(ind)]

# *****************************
# **** PLOT AXIS FUNCTIONS ****
# *****************************
class Figure():
	def __init__(self,nrows=1,ncols=1,sharex=False,sharey=False,squeeze=False,figsize=(8,6)):
		(self.f,self.ax) = plt.subplots(nrows=nrows,ncols=ncols,sharex=sharex,sharey=sharey,squeeze=squeeze)
		if(~(squeeze*(nrows*ncols==1))):
			self.ax = self.ax.flatten();

		self.M = nrows
		self.N = ncols
		n = nrows*ncols
		[xs,ys] = np.meshgrid(np.arange(0,ncols),np.arange(0,nrows))
		self.is_bot = ys.flatten()==nrows-1
		self.is_top = ys.flatten()==0
		self.is_left = xs.flatten()==0
		self.is_right = xs.flatten()==ncols-1

		figsize = np.array(figsize)
		self.f.set_size_inches(figsize[0],figsize[1],forward=True)
		# self.f.set_figheight[figsize[0]]
		# self.f.set_figwidth[figsize[1]]

	def axis(self,cmd):
		for ax in self.ax:
			ax.axis(cmd)

	def set_xlabel(self,lbl):
		for (aa,ax) in enumerate(self.ax):
			if(self.is_bot[aa]):
				ax.set_xlabel(lbl)

	def set_ylabel(self,lbl):
		for (aa,ax) in enumerate(self.ax):
			if(self.is_left[aa]):
				ax.set_ylabel(lbl)

def figure(nrows=1,ncols=1):
	# INPUTS
	# 	nrows
	# 	ncols
	# OUTPUTS - returned
	#  	
	(h,ax) = plt.subplots(nrows,ncols,squeeze=False)
	ax = ax.flatten()
	return (h,ax)

def set_ax_labels(ax, strs, axis=0, rotation=0, axis_inds=-1):
	if(check_value(axis_inds,-1)):
		axis_inds = axis_inds = np.arange(0,strs.size)
	if(axis==0):
		ax.set_xticks(axis_inds)
		ax.set_xticklabels(strs, rotation=rotation)
	elif(axis==1):
		ax.set_yticks(axis_inds)
		ax.set_yticklabels(strs, rotation=rotation)

def set_labels(handles, strs):
	for ss in range(strs.size):
		handles[ss].set_label(strs[ss])

def ax_imshow(ax,mtx,xlim=-1,ylim=-1,title=-1,extent=-1,clim=-1,xlabel=-1,ylabel=-1,xvis=False,yvis=False):
	# INPUTS
	# 	ax 			axis handle 		
	# 	mtx 		2d or 3d array 		
	# 	xlim,ylim 	2 tuple 			
	# 	title 		string 				
	# 	extent 		4 tuple 			
	# 	clim 		2 tuple 			
	ax.cla()
	if(check_value(extent,-1)):
		extent = [0,mtx.shape[0],mtx.shape[1],0]

	if(check_value(clim,-1)):
		im = ax.imshow(mtx,extent=extent,interpolation='nearest')
	else:
		im = ax.imshow(mtx,extent=extent,clim=clim,interpolation='nearest')

	if(not check_value(xlim,-1)):
		ax.set_xlim(xlim)
	if(not check_value(ylim,-1)):
		ax.set_ylim(ylim)
	if(not check_value(xlabel,-1)):
		ax.set_xlabel(xlabel)
	if(not check_value(ylabel,-1)):
		ax.set_ylabel(ylabel)
	if(not check_value(title,-1)):
		ax.set_title(title)
	ax.get_xaxis().set_visible(xvis)
	ax.get_yaxis().set_visible(yvis)

	return im

def check_value(inds, val):
	# Check to see if an array is a single element equaling a particular value
	# Good for pre-processing inputs in a function
	if(np.array(inds).size==1):
		if(inds==val):
			return True
	return False

def find(mask):
	# INPUTS
	# 	mask 		N-d array of bools
	# OUTPUTS - returned
	# 	[0] 		M array 	linear position of M Trues
	mask = np.array(mask).flatten()
	return np.where(mask)[0]

def find_nd(mask):
	# INPUTS
	# 	mask 		N-d array of bools
	# OUTPUTS - returned
	# 	[0] 		MxN array 	M Trues in N-d array
	mask = np.array(mask!=0)
	N = mask.ndim
	M = np.sum(mask)
	ret_val = np.zeros((M,N),dtype='uint')
	temp = np.where(mask)
	for nn in range(N):
		ret_val[:,nn] = temp[nn]
	return ret_val

def meshgrid_subs(xs,ys=np.Infinity,useTril=False,remDiag=False):
	# INPUTS
	# 	xs 			X array
	# 	ys 			Y array 	[Infinity]
	# 	useTril 	bool 		whether or not to use only the lower triangle
	# 	remDiag 	bool 		whether or not to use diagonal, only really takes effect if xs and ys are of equal length
	# OUTPUTS - returned
	#	[0]			XYx2
	ys = np.array(ys)
	if(check_value(ys,np.Infinity)): # only using a single value
		ys = xs

	X = xs.size
	Y = ys.size
	[xs,ys] = np.meshgrid(xs,ys)
	mask = np.ones_like(xs,dtype='bool')

	if(useTril):
		mask = np.tril(mask)
	if(remDiag and (X==Y)):
		mask[np.arange(X),np.arange(Y)] = False
	mask = mask.flatten()
	xs = xs.flatten()[mask]
	ys = ys.flatten()[mask]

	return np.c_[[xs,ys]].transpose(1,0)

def mask_to_inds(mask):
	# If an array of bools, return positions
	# INPUTS
	# 	mask 		N bools or M array
	# OUTPUTS - returned
	# 	[0] 		M array, where M is number of Trues in boolean input

	mask = np.array(mask)

	# if input is a boolean, return its mask
	if(mask.dtype==np.bool):
		return np.where(mask)[0]
	else:
		return mask

def inds_to_mask(inds,N):
	# INPUTS
	# 	
	mask = np.zeros(N, dtype='bool')
	mask[inds] = True
	return mask

def printV(text, obj):
	if(obj.verbose):
		print(text)

# HDF5 file
# def load_from_hdf5(HDF5_FILEPATH, output=-1, prefix='', suffix=''):
# 	# load an hdf5 file and put all of its keys into a dict object
# 	# INPUTS
# 	# 	HDF5_FILEPATH 		string 		filepath of hdf5 file to load
# 	#	output 				dict 		dictionary with elements to load
# 	# 	prefix				string 		prefix to add to key names
# 	# 	suffix 				string 		suffix to add to key names
# 	if(os.path.exists(HDF5_FILEPATH)):
# 		h5_file = h5py.File(HDF5_FILEPATH, 'r')
# 		if(check_value(output,-1)):
# 			output = {} # initialize
# 		for feat in h5_file.keys():
# 			output[prefix+feat+suffix] = h5_file[feat]
# 		return output
# load from hdf5, get away from ordering problem

def auc(xs,ys): # use reimann sums to get area under the curve
	return np.sum((ys[1:]+ys[0:-1])/2 * (xs[1:]-xs[0:-1]))

def check_overwrite(filepath, overwrite, verbose=False):
	# Check if a file exists. If it does, overwrite it depending on input
	# INPUTS
	# 	filepath 		string 		filepath to look for
	# 	overwrite 		bool 		whether or not to overwrite it if the file already exists
	# 	verbose 		string 		[False]
	# OUTPUTS - returned
	# 	[0] 			bool 		False if filepath already exists and overwrite is set to False.
	# 								True to proceed to overwrite
	if(os.path.exists(filepath)):
		if(overwrite):
			os.remove(filepath)
			TO_WRITE = True
			if(verbose):
				print("Overwriting existing file: "+str(filepath))
		else:
			TO_WRITE = False
			if(verbose):
				print("File already exists: "+str(filepath))
	else:
		TO_WRITE = True
		if(verbose):
			print("File does not exist, creating: "+str(filepath))

	return TO_WRITE

def label_sort(labels):
	L = np.max(labels)+1
	cnts = np.histogram(labels, np.arange(-.5,L+.5,1))[0]
	inds0 = np.argsort(cnts)[::-1]
	inds1 = np.argsort(inds0)
	return inds1[labels]

def kmeans_sort(dists, km, returnCnts=False):
	N = km.labels_.size
	K = km.n_clusters
	cnts = np.histogram(km.labels_, np.arange(-.5,km.n_clusters+.5,1))[0]
	inds0 = np.argsort(cnts)[::-1]
	inds1 = np.argsort(inds0)
	km.labels_ = inds1[km.labels_] # shuffle indices
	km.cluster_centers_ = km.cluster_centers_[inds0,:]
	dists = dists[:,inds0]

	if(returnCnts):
		cnts = np.histogram(km.labels_, np.arange(-.5,K+.5,1))[0]
		return (dists,cnts)
	else:
		return dists

def read_csv(filepath, removeEmpty=True):
	csv_fopen = open(filepath,'r')
	csv_obj = csv.reader(csv_fopen)
	rows = np.array([np.array(row) for row in csv_obj])
	if(removeEmpty):
		for (rr,row) in enumerate(rows):
			rows[rr] = rows[rr][row!='']
	return rows

class KMeansLogger():
	# Log k-means results
	def __init__(self, dists, cnts, labels, cluster_centers):
		(self.N, self.K) = dists.shape
		self.dists = dists
		self.cnts = cnts
		self.labels = labels
		self.cluster_centers = cluster_centers
		self.dists_to_cent = self.dists[np.arange(0,self.N),self.labels]
		self.sorted_inds = {}
		self.Nk = np.zeros(self.K, dtype=np.uint)
		for kk in range(self.K):
			mask = self.labels==kk
			inds = find(mask)
			self.sorted_inds[kk] = inds[np.argsort(self.dists_to_cent[mask])]
			self.Nk[kk] = self.sorted_inds[kk].size

	def save(self, filepath):
		np.savez(filepath, self.dists, self.cnts, self.labels, self.cluster_centers,
			self.N, self.K, self.dists_to_cent, self.sorted_inds, self.Nk)

class ConfMtx():
	def __init__(self,gt,cl,specEdges=False,edgesGt=-1,edgesCl=-1,C=-1):
		# INPUTS
		# 	gt 			array of labels, will be flattened
		# 	cl 			array of labels, will be flattened
		# 	specEdges 	[False] go from specified edges or by indices
		# 	edgesGt 	[-1] edges for ground truth values, could be continuous
		# 	edgesCl 	[-1] edges of classification values, could be continuous
		# 	C 			[-1] total number of classes

		# Preprocessing to find edges
		if(specEdges): # edges are either already specified, or we have them
			spec_gt = ~check_value(edgesGt,-1)
			spec_cl = ~check_value(edgesCl,-1)
			if(spec_gt and ~spec_cl): # use same edges
				self.edgesGt = edgesGt
				self.edgesCl = edgesGt
			elif(~spec_gt and spec_cl): # use same edges
				self.edgesGt = edgesCl
				self.edgesCl = edgesCl
			elif(spec_gt and spec_cl): # both specified
				self.edgesGt = edgesGt
				self.edgesCl = edgesCl
			elif(~spec_gt and ~spec_cl): # both specified
				edgeMin = np.minimum(np.min(cl),np.min(gt))
				edgeMax = np.maximum(np.max(cl),np.max(gt))
				edges = np.linspace(edgeMin,edgeMax,num=10,endpoint=True)
				self.edgesGt = edges
				self.edgesCl = edges
		else:
			spec_C = ~check_value(C,-1)
			if(~spec_C):
				spec_C = np.maximum(np.max(gt),np.max(cl))
			self.edgesGt = np.arange(-.5,C+1.5)
			self.edgesCl = self.edgesGt

		# Bin midpoints
		self.midGt = .5*(self.edgesGt[:-1]+self.edgesGt[1:])
		self.midCl = .5*(self.edgesCl[:-1]+self.edgesCl[1:])

		# print self.edgesGt
		# print self.edgesCl
		# print 

		# gt vs cl matrices
		self.CM_cnt = np.histogram2d(gt.flatten(),cl.flatten(),bins=(self.edgesGt,self.edgesCl))[0]
		self.CM_rec = self.CM_cnt/np.sum(self.CM_cnt,axis=1)[:,na()]

# class Radial2DEncoder():
	# def __init__()

class Grid2DEncoder():
	# Encode 2d points as a linear combination of points on a 2D grid
	def __init__(self,amin,amax,ainc,bmin,bmax,binc,verbose=False):
		# OUTPUTS
		# 	ab_grid 		AxBx2
		# 	ab_grid_flt 	ABx2 	increment b first, then a

		self.amin = amin
		self.amax = amax
		self.ainc = ainc
		self.bmin = bmin
		self.bmax = bmax
		self.binc = binc
		self.a_vals = np.arange(amin,amax,ainc) # array of indices
		self.b_vals = np.arange(bmin,bmax,binc)
		self.a_vals_edge = np.arange(amin-ainc/2.,amax+ainc/2.,ainc) # array of indices
		self.b_vals_edge = np.arange(bmin-binc/2.,bmax+binc/2.,binc)
		self.A = self.a_vals.size
		self.B = self.b_vals.size
		self.AB = self.A*self.B

		(a_grid, b_grid) = np.meshgrid(self.a_vals,self.b_vals,indexing='ij')
		a_grid = a_grid.flatten()[:,na()]
		b_grid = b_grid.flatten()[:,na()]
		self.ab_grid_flat = np.concatenate((a_grid,b_grid),axis=1) # ABx2 grid indices		
		self.ab_grid = self.ab_grid_flat.reshape((self.A,self.B,2)) # AxBx2 grid indices

		(a_grid, b_grid) = np.meshgrid(np.arange(0,self.A),np.arange(0,self.B),indexing='ij')
		a_grid_inds = a_grid.flatten()[:,na()]
		b_grid_inds = b_grid.flatten()[:,na()]
		self.ab_grid_inds_flat = np.concatenate((a_grid_inds,b_grid_inds),axis=1) # ABx2 grid indices		
		self.ab_grid_inds = self.ab_grid_inds_flat.reshape((self.A,self.B,2)) # AxBx2 grid indices

		# self.a_grid, self.b_grid)
		# (ab_grid['a_grid'], ab_grid['b_grid']) = np.meshgrid(ab_grid['a_inds'],ab_grid['b_inds'])
		# (ab_grid['a_grid_inds'], ab_grid['b_grid_inds']) = np.meshgrid(np.arange(ab_grid['A']),np.arange(ab_grid['B']))
		# ab_grid['ab_grid'] = np.concatenate((ab_grid['a_grid'].flatten()[:,rz.na()], ab_grid['b_grid'].flatten()[:,rz.na()]),axis=1)
		# ab_grid['AB'] = ab_grid['A']*ab_grid['B']

	def encode_points_mtx_nd(self,pts_nd,axis=1,returnFlat=False):
		# INPUTS
		# 	pts_nd 			nd array
		# 	axis 			integer
		# 	returnFlat 		bool 		return in a flattened array

		pts_flt = flatten_nd_array(pts_nd,axis=axis)
		pts_enc_flt = self.encode_points(pts_flt,returnMatrix=True)
		if(returnFlat):
			return pts_enc_flt
		else:
			return unflatten_2d_array(pts_enc_flt,pts_nd,axis=axis)

	def decode_points_mtx_nd(self,pts_enc_nd,axis=1):
		pts_enc_flt = flatten_nd_array(pts_enc_nd,axis=axis)
		pts_dec_flt = self.decode_points(pts_enc_flt)
		return unflatten_2d_array(pts_dec_flt,pts_enc_nd,axis=axis)

	def encode_points(self,pts,returnMatrix=False):
		# Encode an array of points, either as a matrix or a list of indices and weights
		# INPUTS
		# 	pts 			Nx2 array of coordinates
		# 	returnMatrix 	[false] if True, return full matrix
		# OUTPUTS
		# 	if returnMatrix is True
		# 			NxAB array, rows sum up to one
		# 	if returnMatrix is False
		# 	[0]		Nx4 array 	indices
		# 	[1] 	Nx4 array 	weights

		# find 2 closest indices in each dimension, along with thie rlinear weights
		N = pts.shape[0]
		pts_a = pts[:,0]
		pts_b = pts[:,1]
		
		# index positions (soft)
		pts_a_norm = 1.*(pts_a-self.amin)/self.ainc
		pts_b_norm = 1.*(pts_b-self.bmin)/self.binc

		# index positions, min and max (hard)
		# assumes we never hit the max value
		subs_abs_minmax = np.zeros((N,2,2),dtype='int')
		# index0 is over points
		# index1 is min or max
		# index2 is coordinate a or b
		subs_abs_minmax[:,0,0] = np.floor(pts_a_norm)
		subs_abs_minmax[:,1,0] = np.floor(pts_a_norm)+1
		subs_abs_minmax[:,0,1] = np.floor(pts_b_norm)
		subs_abs_minmax[:,1,1] = np.floor(pts_b_norm)+1

		# 4 coordinates
		subs_abs = np.zeros((N,4,2))
		# index1, a0b0, a0b1, a1b0, a1b1
		# index2, a, b
		subs_abs[:,0,0] = subs_abs_minmax[:,0,0] # a0b0
		subs_abs[:,0,1] = subs_abs_minmax[:,0,1] # a0b0
		subs_abs[:,1,0] = subs_abs_minmax[:,0,0] # a0b1
		subs_abs[:,1,1] = subs_abs_minmax[:,1,1] # a0b1
		subs_abs[:,2,0] = subs_abs_minmax[:,1,0] # a1b0
		subs_abs[:,2,1] = subs_abs_minmax[:,0,1] # a1b0
		subs_abs[:,3,0] = subs_abs_minmax[:,1,0] # a1b0
		subs_abs[:,3,1] = subs_abs_minmax[:,1,1] # a1b0
		inds_abs = sub2ind2(subs_abs.reshape((N*4,2)), (self.A,self.B)).reshape((N,4))

		# assign weights to indices
		wts_a = np.zeros((N,2),dtype='float32') # index1 weight on min,max
		wts_a[:,0] = (1-(pts_a_norm-subs_abs_minmax[:,0,0]))
		wts_a[:,1] = 1-wts_a[:,0]

		wts_b = np.zeros((N,2),dtype='float32') # index1 weight on min,max
		wts_b[:,0] = (1-(pts_b_norm-subs_abs_minmax[:,0,1]))
		wts_b[:,1] = 1-wts_b[:,0]

		# convert subscripts to indices
		wts_abs = wts_a[:,:,na()] * wts_b[:,na(),:]
		wts_abs = wts_abs.reshape((N,4))

		if(returnMatrix):
			# convert to expanded version
			wts_abs_exp = np.zeros((N,self.AB),dtype='float32')
			for nn in range(N):
				# wts_abs_exp[nn,inds_abs[nn,:]] = wts_abs[nn,:]
				wts_abs_exp[nn,inds_abs[nn,:]] = wts_abs[nn,:]
				# print inds_abs[nn,:]
				# print wts_abs[nn,:]
			return wts_abs_exp
		else:
			sort_inds = np.argsort(wts_abs,axis=1)[:,::-1]
			static_inds = np.ogrid[:N,0:4]
			wts_abs = wts_abs[static_inds[0],sort_inds]
			inds_abs = inds_abs[static_inds[0],sort_inds]

			return(inds_abs,wts_abs)

	def decode_points(self,pts_enc):
		# INPUTS
		# 	pts_enc		NxAB array, rows sum up to one
		# OUTPUTS
		# 				Nx2 array of coordinates
		return np.dot(pts_enc, self.ab_grid_flat)

	def decode_points_wts(self,inds_abs,wts_abs=-1):
		# INPUTS
		# 	inds_abs 	NxM 	indices into the grid
		# 	wts_abs 	NxM 	weights into the grid
		#  -- or --
		# 	inds_abs 	N 		single indices into grid
		# 	wts_abs 	[] 		not provided 
		# CONSTANTS
		# 	N 	number of points
		# 	M 	number of vertices
		# OUTPUTS
		# 				Nx2
		if(check_value(wts_abs,-1)):
			inds_abs = inds_abs[:,na()]
			wts_abs = np.ones_like(inds_abs)

		return np.sum(self.ab_grid_flat[inds_abs,:] * wts_abs[:,:,na()],axis=1)

	def encode_nearest_points(self,pts):
		# INPUTS
		# 	pts 		Nx2 	points in space
		# OUTPUTS
		#	inds 		N 		indices into grid
		pts_norm = np.round((pts-np.array((self.amin,self.bmin))[na(),:])/np.array((self.ainc,self.binc))[na(),:])
		N = pts_norm.shape[0]
		return sub2ind2(pts_norm.reshape((N,2)),(self.A,self.B)).reshape((N))

def reshape_single_axis(in_pts,A,B,axis=0):
	''' Reshape a single axis to a 2d in an nd array
	INPUTS
		in_pts 		nd array
		A,B 		
	 	axis'''
	NDIM = in_pts.ndim
	SHP = np.array(in_pts.shape)
	nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
	NPTS = np.prod(SHP[nax])
	axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)

	NDIMnew = np.zeros(NDIM+1,dtype='int')
	# print NDIMnew
	NDIMnew[:axis] = SHP[:axis]
	# print NDIMnew
	NDIMnew[axis] = A
	# print NDIMnew
	NDIMnew[axis+1] = B
	# print NDIMnew
	# print axis
	# print NDIM
	if(axis<NDIM-1):
		NDIMnew[axis+2:] = SHP[axis+1:]
		# print NDIMnew

	return np.reshape(in_pts,NDIMnew)

def flatten_nd_array(pts_nd,axis=1):
	# Flatten an nd array into a 2d array with a certain axis
	# INPUTS
	# 	pts_nd 		N0xN1x...xNd array
	# 	axis 		integer
	# OUTPUTS
	# 	pts_flt 	prod(N \ N_axis) x N_axis array
	NDIM = pts_nd.ndim
	SHP = np.array(pts_nd.shape)
	nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
	NPTS = np.prod(SHP[nax])
	axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
	pts_flt = pts_nd.transpose((axorder))
	pts_flt = pts_flt.reshape(NPTS,SHP[axis])
	return pts_flt

def unflatten_2d_array(pts_flt,pts_nd,axis=1,squeeze=False):
	# Unflatten a 2d array with a certain axis
	# INPUTS
	# 	pts_flt 	prod(N \ N_axis) x M array
	# 	pts_nd 		N0xN1x...xNd array
	# 	axis 		integer
	# 	squeeze 	bool 	if true, M=1, squeeze it out
	# OUTPUTS
	# 	pts_out 	N0xN1x...xNd array	
	NDIM = pts_nd.ndim
	SHP = np.array(pts_nd.shape)
	nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
	NPTS = np.prod(SHP[nax])

	if(squeeze):
		axorder = nax
		axorder_rev = np.argsort(axorder)
		M = pts_flt.shape[1]
		NEW_SHP = SHP[nax].tolist()
		# print NEW_SHP
		# print pts_flt.shape
		pts_out = pts_flt.reshape(NEW_SHP)
		pts_out = pts_out.transpose(axorder_rev)
	else:
		axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
		axorder_rev = np.argsort(axorder)
		M = pts_flt.shape[1]
		NEW_SHP = SHP[nax].tolist()
		NEW_SHP.append(M)
		pts_out = pts_flt.reshape(NEW_SHP)
		pts_out = pts_out.transpose(axorder_rev)

	return pts_out

def grid_ab(inc=10,min_val=-110):
	# Define a grid over ab coordinates
	amin = min_val
	bmin = min_val
	amax = -min_val+inc
	bmax = -min_val+inc
	return Grid2DEncoder(amin,amax,inc,bmin,bmax,inc)

def load_kmeans_log(filepath):
	load_vars = np.load(filepath)
	return KMeansLogger(load_vars['arr_0'],load_vars['arr_1'],load_vars['arr_2'],load_vars['arr_3'])

def load_npz_dict(filepath, names, returnTuple=True):
	# Load compressed numpy file into a dictionary or a tuple
	# INPUTS
	# 	filepath 		string
	#	names 			string array
	# 	returnTuple 	bool 			[True] return as tuple, or return as dictoinary
	# OUTPUTS - returned
	# 	[ ] 			list or dict 	
	
	load_data = np.load(filepath)
	out_dict = {}
	for (nn,name) in enumerate(names):
		out_dict[name] = load_data['arr_%i'%nn]

	if(returnTuple):
		out_tuple = []
		for (nn,name) in enumerate(names):
			out_tuple.append(out_dict[name])
		return out_tuple

	return out_dict

def str_max_len(strs, returnStr=False):
	strs = np.array(strs)
	maxL = np.max(np.array([len(in_str) for in_str in strs.flatten()]))
	if(returnStr):
		return 'S'+str(maxL)
	else:
		return maxL

def print_strs(in_strs):
	for cur_str in in_strs:
		print(cur_str)

def print_str_val(in_strs,in_vals,inds=-1,str_format='%.3f'):
	# INPUTS
	# 	in_strs 		S strings
	# 	in_vals 		S values
	# 	I 				scalar or indices
	# 	str_format 		string 			amount to format
	if(check_value(inds,-1)):
		inds = np.arange(in_strs.size)
	for (iii,ii) in enumerate(inds):
		print(('%i: '+str_format+' %s')%(ii,in_vals[ii],in_strs[ii]))

# def print_caffe_shapes(net,layers=-1,first_layer=-1,last_layer=-1):
	# caffe_shapes(net,layers=layers,first_layer=first_layer,last_layer=last_layer):

def caffe_shapes(net,layers=-1,first_layer=-1,last_layer=-1,to_print=False):
	if(check_value(layers,-1)):
		layers = net._blob_names

	START_FLAG = False
	if(check_value(first_layer,-1)):
		START_FLAG = True

	num_params = np.zeros(len(layers))
	for (ll,layer) in enumerate(layers):
		if(START_FLAG or (layer==first_layer)):
			START_FLAG = True

			if(layer==last_layer):
				START_FLAG = False

		if(START_FLAG):
			if('split' not in layer):
				shp = net.blobs[layer].data.shape
				# if(len(shp)==2):
				# 	print '%ix%i (%s)'%(shp[0],shp[1],layer)
				# elif(len(shp)==4):
				# 	print '%ix%ix%ix%i (%s)'%(shp[0],shp[1],shp[2],shp[3],layer)
				shp_str = ''
				for s in shp:
					shp_str = shp_str+'x'+str(s)
				shp_str = shp_str[1:]
				if(to_print):
					print('%s: \t %s'%(shp_str,layer))

				num_params[ll] = np.prod(np.array(shp))

	return num_params

# def print_caffe_param_shapes(net,layers=-1,first_layer=-1,last_layer=-1):
# 	caffe_param_shapes(net,layers=layers,first_layer=first_layer,last_layer=last_layer):

def caffe_param_shapes(net,layers=-1,first_layer=-1,last_layer=-1,first_blob_only=True,to_print=False):
	num_params = []
	if(check_value(layers,-1)):
		layers = net.params.keys()

	START_FLAG = False
	if(check_value(first_layer,-1)):
		START_FLAG = True

	for (ll,layer) in enumerate(layers):
		if(START_FLAG or (layer==first_layer)):
			START_FLAG = True

			if(layer==last_layer):
				START_FLAG = False

		if(START_FLAG):
			# shp = net.blobs[layer].data.shape
			# if(len(shp)==2):
			# 	print '%ix%i (%s)'%(shp[0],shp[1],layer)
			# elif(len(shp)==4):
			# 	print '%ix%ix%ix%i (%s)'%(shp[0],shp[1],shp[2],shp[3],layer)
			for pp in range(len(net.params[layer])):
				shp = np.array(net.params[layer][pp].data.shape)
				shp_str = ''
				if((not first_blob_only) or pp==0):
					num_params.append(np.product(shp))
				for s in shp:
					shp_str = shp_str+'x'+str(s)
				shp_str = shp_str[1:]
				if(to_print):
					print('%s [%i]:\t%s'%(layer,pp,shp_str))

	return np.array((num_params))

def cmap_to_color(cmap,bb,B):
	return cmap(1.*bb/(B))

def plot_losses(ax,iters,losses,base_names,set_names,\
	cmap=plt.cm.hsv_r,set_lines='-',inds=0,mults=1,toNorm=False):
	B = base_names.size
	base_names = np.array(base_names).flatten()
	set_names = np.array(set_names).flatten()
	inds = rz.scalar_to_array(B,inds)
	mults = rz.scalar_to_array(B,mults)
	for (bb,base_name) in enumerate(base_names):
		for (ss,set_name) in enumerate(set_names):
			ax.plot(iters[base_name][set_name],mults[bb]*losses[base_name][set_name][:,inds[bb]],\
				set_lines[ss],color=cmap_to_color(cmap,bb,B),\
				linewidth=2,label='%s-%s'%(base_name,set_name),toNorm=toNorm)

def plot_losses_single(ax,iters,losses,set_names,\
	cmap=plt.cm.hsv_r,set_lines='-',chars='',toNorm=False):
	for (ss,set_name) in enumerate(set_names):
		I = losses[set_name].shape[1]
		chars_use = rz.scalar_to_array(I,chars)
		for ii in range(I):
			if(toNorm):
				plot_vals = losses[set_name][:,ii]/(losses[set_name][-1,ii])
			else:
				plot_vals = losses[set_name][:,ii]
			ax.plot(iters[set_name],plot_vals,\
				set_lines[ss],color=cmap_to_color(cmap,ii,I),\
				linewidth=2,label='[%i]-%s-%s'%(ii,set_name,chars_use[ii]))


def softmax_mtx_nd(mtx_nd,axis=1):
	mtx_flt = flatten_nd_array(mtx_nd,axis=axis)
	mtx_flt = mtx_flt - np.max(mtx_flt,axis=1)[:,na()]
	# print mtx_flt
	sm_flt = np.exp(mtx_flt)/np.sum(np.exp(mtx_flt),axis=1)[:,na()]
	return unflatten_2d_array(sm_flt,mtx_nd,axis=axis)

def sort_inds(orig_inds):
	# sort indices, and then provide indices to unravel the sorted indices
	# orig_inds = sorted_inds[untangle_inds]

	sorted_inds = np.sort(orig_inds)
	untangle_inds = np.argsort([np.argsort(orig_inds)])
	return (sorted_inds, untangle_inds)

def scalar_to_array(N,inp_arr):
	''' Check if an variable is an array or a scalar. If it's a scalar, expand
	it to the specified size. If not, return it.
	INPUTS
		N 			number of blocks
		inp_arr 	N numpy array or single value
	OUTPUTS
	 	out_arr 	N numpy array
	'''
	tmp = np.array(inp_arr)
	if(tmp.size==1):
		out_array = np.zeros(N,dtype=tmp.dtype)
		out_array[...] = inp_arr
		return out_array
	else:
		return inp_arr # return input array