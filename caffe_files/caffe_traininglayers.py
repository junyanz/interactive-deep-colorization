# **************************************
# ***** Richard Zhang / 2016.08.06 *****
# **************************************
import numpy as np
import warnings
import os
import sklearn.neighbors as nn
import caffe
from skimage import color
import matplotlib.pyplot as plt
import math
import platform
import cv2
import rz_fcns_nohdf5 as rz

# ***************************************
# ***** LAYERS FOR GLOBAL HISTOGRAM *****
# ***************************************
class SpatialRepLayer(caffe.Layer):
    '''
    INPUTS
        bottom[0].data  NxCx1x1
        bottom[1].data  NxCxXxY
    OUTPUTS
        top[0].data     NxCxXxY     repeat 0th input spatially  '''
    def setup(self,bottom,top):
        if(len(bottom)!=2):
            raise Exception("Layer needs 2 inputs")

        self.param_str_split = self.param_str.split(' ')
        # self.keep_ratio = float(self.param_str_split[0]) # frequency keep whole input

        self.N = bottom[0].data.shape[0]
        self.C = bottom[0].data.shape[1]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]

        if(self.X!=1 or self.Y!=1):
            raise Exception("bottom[0] should have spatial dimensions 1x1")

        # self.Nref = bottom[1].data.shape[0]
        # self.Cref = bottom[1].data.shape[1]
        self.Xref = bottom[1].data.shape[2]
        self.Yref = bottom[1].data.shape[3]

    def reshape(self,bottom,top):
        top[0].reshape(self.N,self.C,self.Xref,self.Yref) # output shape

    def forward(self,bottom,top):
        top[0].data[...] = bottom[0].data[:,:,:,:] # will do singleton expansion

    def backward(self,top,propagate_down,bottom):
        bottom[0].diff[:,:,0,0] = np.sum(np.sum(top[0].diff,axis=2),axis=2)
        bottom[1].diff[...] = 0

class ColorGlobalDropoutLayer(caffe.Layer):
    '''
    Inputs
        bottom[0].data      NxCx1x1
    Outputs
        top[0].data         Nx(C+1)x1x1     last channel is whether or not to keep input
                                            first C channels are copied from bottom (if kept)
    '''
    def setup(self,bottom,top):
        if(len(bottom)==0):
            raise Exception("Layer needs inputs")   

        self.param_str_split = self.param_str.split(' ')
        self.keep_ratio = float(self.param_str_split[0]) # frequency keep whole input
        self.cnt = 0

        self.N = bottom[0].data.shape[0]
        self.C = bottom[0].data.shape[1]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]

    def reshape(self,bottom,top):
        top[0].reshape(self.N,self.C+1,self.X,self.Y) # output mask

    def forward(self,bottom,top):
        top[0].data[...] = 0
        # top[0].data[:,:self.C,:,:] = bottom[0].data[...]

        # determine which ones are kept
        keeps = np.random.binomial(1,self.keep_ratio,size=self.N)

        top[0].data[:,-1,:,:] = keeps[:,np.newaxis,np.newaxis]
        top[0].data[:,:-1,:,:] = bottom[0].data[...]*keeps[:,np.newaxis,np.newaxis,np.newaxis]

    def backward(self,top,propagate_down,bottom):
        0; # backward not implemented

class ChooseOneDropoutLayer(caffe.Layer):
    '''
    Inputs
        bottom[0].data      NxCx1x1
    Outputs
        top[0].data         Nx2Cx1x1    evens are the bottom data (0)
                                        odds indicate which one is kept
    '''
    def setup(self,bottom,top):
        if(len(bottom)==0):
            raise Exception("Layer needs inputs")   

        self.param_str_split = self.param_str.split(' ')
        self.drop_all_ratio = float(self.param_str_split[0]) # frequency keep whole input
        self.cnt = 0

        self.N = bottom[0].data.shape[0]
        self.C = bottom[0].data.shape[1]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]

    def reshape(self,bottom,top):
        top[0].reshape(self.N,2*self.C,self.X,self.Y) # output mask

    def forward(self,bottom,top):
        top[0].data[...] = 0 # clear everything

        # determine which ones are kept
        drop_alls = np.random.binomial(1,self.drop_all_ratio,size=self.N)

        # determine which to keep
        keep_inds = np.random.randint(self.C,size=self.N)

        for nn in range(self.N):
            if(drop_alls[nn]==0):
                keep_ind = keep_inds[nn]
                top[0].data[nn,2*keep_ind,0,0] = bottom[0].data[nn,keep_ind,0,0]
                top[0].data[nn,2*keep_ind+1,0,0] = 1
            # top[0].data[:,-1,:,:] = keeps[:,np.newaxis,np.newaxis]

    def backward(self,top,propagate_down,bottom):
        0; # backward not implemented


# **************************************
# ***** RANDOM REVEALING OF COLORS *****
# **************************************
class ColorRandPointLayer(caffe.Layer):
    ''' Layer which reveals random square chunks of the input color '''
    def setup(self,bottom,top):
        if(len(bottom)==0):
            raise Exception("Layer needs inputs")   

        self.param_str_split = self.param_str.split(' ')
        self.cnt = 0

        self.mask_mult = 110.

        self.N = bottom[0].data.shape[0]
        self.C = bottom[0].data.shape[1]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]

        self.p_numpatch = 0.125 # probability for number of patches to use drawn from geometric distribution
        self.p_min_size = 0 # half-patch min size
        self.p_max_size = 4 # half-patch max size
        self.p_std = .25 # percentage of image for std where patch is located
        self.p_whole = .01 # probability of revealing whole image

    def reshape(self,bottom,top):
        top[0].reshape(self.N,self.C+1,self.X,self.Y) # output mask

    def forward(self,bottom,top):
        top[0].data[...] = 0
        # top[0].data[:,:self.C,:,:] = bottom[0].data[...]

        # determine number of points
        Ns = np.random.geometric(p=self.p_numpatch,size=self.N)

        # determine half-patch sizes
        Ps = np.random.random_integers(self.p_min_size,high=self.p_max_size,size=np.sum(Ns))

        #determine location
        Xs = np.clip(np.random.normal(loc=self.X/2.,scale=self.X*self.p_std,size=np.sum(Ns)),0,self.X)
        Ys = np.clip(np.random.normal(loc=self.Y/2.,scale=self.Y*self.p_std,size=np.sum(Ns)),0,self.Y)

        use_wholes = np.random.binomial(1,self.p_whole,size=self.N)

        cnt = 0
        for nn in range(self.N):
            if(use_wholes[nn]==1): # throw in whole image
                # print('Using whole image')
                top[0].data[nn,:self.C,:,:] = bottom[0].data[nn,:,:,:]
                top[0].data[nn,-1,:,:] = self.mask_mult
                cnt = cnt+Ns[nn]
            else: # sample points
                for nnn in range(Ns[nn]):
                    p = Ps[cnt]
                    x = Xs[cnt]
                    y = Ys[cnt]

                    # print '(%i,%i,%i)'%(x,y,p)
                    top[0].data[nn,:self.C,x-p:x+p+1,y-p:y+p+1] \
                        = np.mean(np.mean(bottom[0].data[nn,:,x-p:x+p+1,y-p:y+p+1],axis=1),axis=1)[:,np.newaxis,np.newaxis]
                    top[0].data[nn,-1,x-p:x+p+1,y-p:y+p+1] = self.mask_mult
                    cnt = cnt+1

    def backward(self,top,propagate_down,bottom):
        0; # backward not implemented

# Randomly reveal strokes
def gen_random_stroke(X,Nmin=0,Nmax=8,Lmin=4,Lmax=20):
    ''' Generate a random stroke 
    (1) Randomly pick a direction and location to begin with
    (2) Randomly choose number of points, loop through points
        (a) randomly generate delta_theta, length
        (b) append to list of points
        (c) if the point crosses the edge, exist loop
    (3) Clip on boundaries and return
    INPUTS
        X      size of image
        Nmin   min number of points
        Nmax   max number of points
        Lmin   min length of a segment
        Lmax   max length of a segment
    '''
    cur_theta = np.random.uniform(-math.pi,math.pi,size=1)

    pts = []
    cur_pt = np.random.uniform(.1*X,.9*X,size=2)
    pts.append(cur_pt.copy())
    
    N = np.random.randint(Nmin,Nmax) # number of points
    dtheta_bnd = np.random.uniform(-.4,.4) # amount that curve will deviate
    for nn in range(N):
        delta_theta = np.random.uniform(-dtheta_bnd*math.pi,dtheta_bnd*math.pi,size=1) # deviation
        cur_length = np.random.uniform(Lmin,Lmax,size=1) # pixels
        
        cur_theta = cur_theta+delta_theta
        
        cur_pt = cur_pt + np.array((cur_length*math.cos(cur_theta),cur_length*math.sin(cur_theta))).flatten()
        pts.append(cur_pt.copy())
        
        if(np.sum(cur_pt<0) or np.sum(cur_pt>X-1)): # went out of bounds
            break
        
    return np.array(np.clip(pts,0,X-1))

def stroke2mask(pts,W,X,returnFlat=False):
    ''' Given stroke endpoints and line thickness, return a mask '''
    pts = pts.astype('int')
    cur_img = np.zeros((X,X,1),dtype='uint8')
    for pp in range(pts.shape[0]-1):
        cur_img = cv2.line(cur_img,(pts[pp,0],pts[pp,1]),(pts[pp+1,0],pts[pp+1,1]),(255,255,255),thickness=W)
    cur_img_mask = cur_img[:,:,0]>0
    cur_img_mask_flt = cur_img_mask.flatten()
    
    if(returnFlat):
        return cur_img_mask_flt
    else:
        return cur_img_mask

class RandStrokePointLayer(caffe.Layer):
    ''' Layer reveals random strokes and points '''
    def setup(self,bottom,top):
        if(len(bottom)==0):
            raise Exception("Layer needs inputs")   

        self.param_str_split = self.param_str.split(' ')
        self.cnt = 0

        self.mask_mult = 110.

        self.N = bottom[0].data.shape[0]
        self.C = bottom[0].data.shape[1]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]

        self.p_numpatch = 0.125 # probability for number of points/strokes to use, drawn from geometric distribution

        self.p_stroke = 0.25 # probability of using a stroke (rather than a point)

        # patch settings
        self.p_min_size = 0 # half-patch min size
        self.p_max_size = 4 # half-patch max size
        self.p_std = .25 # percentage of image for std where patch is located
        self.p_whole = .01 # probability of revealing whole image

        # stroke settings
        self.l_min_thick=1; self.l_max_thick=8; # thickness
        self.l_min_seg=0; self.l_max_seg=10; # number of points per line
        self.l_min_len=0; self.l_max_len=10; # length of each line segment

    def reshape(self,bottom,top):
        top[0].reshape(self.N,self.C+1,self.X,self.Y) # output mask

    def forward(self,bottom,top):
        top[0].data[...] = 0
        # top[0].data[:,:self.C,:,:] = bottom[0].data[...]

        # determine number of points/patches
        Ns = np.random.geometric(p=self.p_numpatch,size=self.N) 
        use_wholes = np.random.binomial(1,self.p_whole,size=self.N)

        # Patch settings
        # determine half-patch sizes
        Ps = np.random.random_integers(self.p_min_size,high=self.p_max_size,size=np.sum(Ns))

        #determine location
        Xs = np.clip(np.random.normal(loc=self.X/2.,scale=self.X*self.p_std,size=np.sum(Ns)),0,self.X)
        Ys = np.clip(np.random.normal(loc=self.Y/2.,scale=self.Y*self.p_std,size=np.sum(Ns)),0,self.Y)

        # stroke or patch
        is_strokes = np.random.binomial(1,self.p_stroke,size=np.sum(Ns))
        Ws = np.random.randint(self.l_min_thick,self.l_max_thick,np.sum(Ns))

        cnt = 0
        for nn in range(self.N):
            if(use_wholes[nn]==1): # throw in whole image
                # print('Using whole image')
                top[0].data[nn,:self.C,:,:] = bottom[0].data[nn,:,:,:]
                top[0].data[nn,-1,:,:] = self.mask_mult
                cnt = cnt+Ns[nn]
            else: # sample points
                for nnn in range(Ns[nn]):
                    if(not is_strokes[nnn]): # point mode
                        p = Ps[cnt]
                        x = Xs[cnt]
                        y = Ys[cnt]

                        # print '(%i,%i,%i)'%(x,y,p)
                        top[0].data[nn,:self.C,x-p:x+p+1,y-p:y+p+1] \
                            = np.mean(np.mean(bottom[0].data[nn,:,x-p:x+p+1,y-p:y+p+1],axis=1),axis=1)[:,np.newaxis,np.newaxis]
                        top[0].data[nn,-1,x-p:x+p+1,y-p:y+p+1] = self.mask_mult
                    else: # stroke mode
                        stroke_pts = gen_random_stroke(self.X,Nmin=self.l_min_seg,Nmax=self.l_max_seg,\
                            Lmin=self.l_min_len,Lmax=self.l_max_len).astype('int')
                        cur_mask = stroke2mask(stroke_pts,Ws[nnn],self.X)

                        cur_mask_inds = rz.find_nd(cur_mask)
                        top[0].data[nn,:self.C,cur_mask_inds[:,0],cur_mask_inds[:,1]] \
                            = bottom[0].data[nn,:,cur_mask_inds[:,0],cur_mask_inds[:,1]]
                        top[0].data[nn,-1,cur_mask_inds[:,0],cur_mask_inds[:,1]] = self.mask_mult

                    cnt = cnt+1

    def backward(self,top,propagate_down,bottom):
        0; # backward not implemented

# **********************************
# ***** PREVIOUSLY MADE LAYERS *****
# **********************************
class DataDropoutLayer(caffe.Layer):
    ''' Layer which drops out chunks of the input '''
    def setup(self,bottom,top):
        if(len(bottom)==0):
            raise Exception("Layer needs inputs")   

        self.param_str_split = self.param_str.split(' ')
        self.dropout_ratio = float(self.param_str_split[0]) # dropout frequency
        self.dropout_size = int(self.param_str_split[1]) # block size for dropout
        self.refresh_period = int(self.param_str_split[2]) # regenerate every few iterations
        self.channel_sync = bool(int(self.param_str_split[3])) # sync dropout through channels

        self.retain_ratio = 1 - self.dropout_ratio

        self.cnt = 0

        self.N = bottom[0].data.shape[0]
        self.C = bottom[0].data.shape[1]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]

        self.Xblock = self.X/self.dropout_size
        self.Yblock = self.Y/self.dropout_size

    def reshape(self,bottom,top):
        top[0].reshape(self.N,self.C,self.X,self.Y) # output mask
        top[1].reshape(self.N,self.C,self.X,self.Y) # masked input

    def forward(self,bottom,top):
        if(np.mod(self.cnt,self.refresh_period)==0):
            if(self.channel_sync):
                retain_block = np.random.binomial(1,self.retain_ratio,size=(self.N,1,self.Xblock,self.Yblock))
            else:
                retain_block = np.random.binomial(1,self.retain_ratio,size=(self.N,self.C,self.Xblock,self.Yblock))
            top[0].data[...] = retain_block.repeat(self.dropout_size,axis=2).repeat(self.dropout_size,axis=3)
        self.cnt = self.cnt+1

        top[1].data[...] = bottom[0].data[...]*top[0].data[...] # mask image

    def backward(self,top,propagate_down,bottom):
        0; # backward not implemented

class LossMeterLayer(caffe.Layer):
    ''' Layer acts as a "meter" to track loss values '''
    def setup(self,bottom,top):
        if(len(bottom)==0):
            raise Exception("Layer needs inputs")

        self.param_str_split = self.param_str.split(' ')
        self.LOSS_DIR = self.param_str_split[0]
        self.P = int(self.param_str_split[1])
        self.H = int(self.param_str_split[2])
        if(len(self.param_str_split)==4):
            self.prefix = self.param_str_split[3]
        else:
            self.prefix = ''
        # self.P = 1000 # interval to print losses
        # self.H = 1000 # history size
        # self.LOSS_DIR = './loss_save'

        self.cnt = 0 # loss track counter
        # self.P = 1 # interval to print losses
        self.h = 0 # index into history
        self.L = len(bottom)
        self.losses = np.zeros((self.L,self.H))

        self.ITER_PATH = os.path.join(self.LOSS_DIR,'iter.npy')
        self.LOG_PATH = os.path.join(self.LOSS_DIR,'loss_log')

        rz.mkdir(self.LOSS_DIR)
        if(os.path.exists(self.ITER_PATH)):
            self.iter = np.load(self.ITER_PATH)
        else:
            self.iter = 0 # iteration counter
        print 'Initial iteration: %i'%(self.iter+1)

    def reshape(self,bottom,top):
        0;
        # top[0].reshape(1)
        # print 'No'

    def forward(self,bottom,top):
        for ll in range(self.L):
            self.losses[ll,self.h] = bottom[ll].data[...]

        if(np.mod(self.cnt,self.P)==self.P-1): # print
            if(self.cnt >= self.H-1):
                tmp_str = 'NumAvg %i, Loss '%(self.H)
                for ll in range(self.L):
                    tmp_str += '%.3f, '%np.mean(self.losses[ll,:])
            else:
                tmp_str = 'NumAvg %i, Loss '%(self.h)
                for ll in range(self.L):
                    tmp_str += '%.3f, '%np.mean(self.losses[ll,:self.cnt+1])
            print_str = '%s: Iter %i, %s'%(self.prefix,self.iter+1,tmp_str)
            print print_str

            self.f = open(self.LOG_PATH,'a')
            self.f.write(print_str)
            self.f.write('\n')
            self.f.close()
            np.save(self.ITER_PATH,self.iter)

        self.h = np.mod(self.h+1,self.H) # roll through history
        self.cnt = self.cnt+1
        self.iter = self.iter+1

    def backward(self,top,propagate_down,bottom):
        for ll in range(self.L):
            continue

class LossMeterLayer(caffe.Layer):
    ''' Layer acts as a "meter" to track loss values '''
    def setup(self,bottom,top):
        if(len(bottom)==0):
            raise Exception("Layer needs inputs")

        self.param_str_split = self.param_str.split(' ')
        self.LOSS_DIR = self.param_str_split[0]
        self.P = int(self.param_str_split[1])
        self.H = int(self.param_str_split[2])
        if(len(self.param_str_split)==4):
            self.prefix = self.param_str_split[3]
        else:
            self.prefix = ''
        # self.P = 1000 # interval to print losses
        # self.H = 1000 # history size
        # self.LOSS_DIR = './loss_save'

        self.cnt = 0 # loss track counter
        # self.P = 1 # interval to print losses
        self.h = 0 # index into history
        self.L = len(bottom)
        self.losses = np.zeros((self.L,self.H))

        self.ITER_PATH = os.path.join(self.LOSS_DIR,'iter.npy')
        self.LOG_PATH = os.path.join(self.LOSS_DIR,'loss_log')

        rz.mkdir(self.LOSS_DIR)
        if(os.path.exists(self.ITER_PATH)):
            self.iter = np.load(self.ITER_PATH)
        else:
            self.iter = 0 # iteration counter
        print 'Initial iteration: %i'%(self.iter+1)

    def reshape(self,bottom,top):
        0;
        # top[0].reshape(1)
        # print 'No'

    def forward(self,bottom,top):
        for ll in range(self.L):
            self.losses[ll,self.h] = bottom[ll].data[...]

        if(np.mod(self.cnt,self.P)==self.P-1): # print
            if(self.cnt >= self.H-1):
                tmp_str = 'NumAvg %i, Loss '%(self.H)
                for ll in range(self.L):
                    tmp_str += '%.3e, '%np.mean(self.losses[ll,:])
            else:
                tmp_str = 'NumAvg %i, Loss '%(self.h)
                for ll in range(self.L):
                    tmp_str += '%.3e, '%np.mean(self.losses[ll,:self.cnt+1])
            print_str = '%s: Iter %i, %s'%(self.prefix,self.iter+1,tmp_str)
            print print_str

            self.f = open(self.LOG_PATH,'a')
            self.f.write(print_str)
            self.f.write('\n')
            self.f.close()
            np.save(self.ITER_PATH,self.iter)

        self.h = np.mod(self.h+1,self.H) # roll through history
        self.cnt = self.cnt+1
        self.iter = self.iter+1

    def backward(self,top,propagate_down,bottom):
        for ll in range(self.L):
            continue

# ***********************************
# ***** PARSE LOSS LOG WRAPPERS *****
# ***********************************
def group_iter_losses(base_dirs,sets,LOSS_ROOTDIR,base_names=-1,set_names=-1,\
    return_min_max=False,min_maxes=1,mask_max=True):
    ''' 
    INPUTS
        base_dirs       base subdirectory to search for loss logs in
        sets            subsubdirectory to find loss log
        LOSS_ROOTDIR    rootdir to attach to all base_dirs
        base_names      [base_dirs] base names to populate dictionary with
        set_names       [set_names] set names to populate dictionary with
        return_min_maxs boolean whether or not to return min/max of dataset
        min_maxs        array of 0/1, 0 for min, 1 for max
    OUTPUTS
        (iters,losses)  
    '''
    base_dirs = np.array(base_dirs)
    sets = np.array(sets)

    B = base_dirs.size
    if(rz.check_value(base_names,-1)):
        base_names = base_dirs
    if(rz.check_value(set_names,-1)):
        set_names = sets
    min_maxes = rz.scalar_to_array(B,min_maxes)

    iters = {}; losses = {}
    if(return_min_max):
        ret_min_maxes = {}
    # if(return_max_iter):
        # max_iter = {}
    for (bb,base) in enumerate(base_dirs):
        base_name = base_names[bb]
        loss_paths = []; names = []
        for (ss,set) in enumerate(sets):
            set_name = set_names[ss]
            loss_paths.append('%s/%s/loss_log'%(base,set))
        if(return_min_max):
            (iters[base_name],losses[base_name],ret_min_maxes[base_name])\
                = parse_loss_logs(set_names,loss_paths,LOSS_ROOTDIR,return_min_max=True,min_maxes=min_maxes,mask_max=mask_max)
        else:
            (iters[base_name],losses[base_name])\
                = parse_loss_logs(set_names,loss_paths,LOSS_ROOTDIR,return_min_max=False,mask_max=mask_max)

    # rets = []
    # if(return_min_max):
    #     rets.append(ret_min_maxes)
    # if(return_max_iter):
    #     rets.append(max_iter)

    # if(return_min_max or return_max_iter):
        # return (iters,losses,rets)
    if(return_min_max):
        return (iters,losses,ret_min_maxes)
    else:
        return (iters,losses)

def parse_loss_logs(names,LOSS_LOG_PATHS,rootdir='',iter_norm_factor=1000,\
    return_min_max=False,min_maxes=1,mask_max=True):
    ''' grab multiple loss_logs'''
    LOSS_LOG_PATHS = np.array(LOSS_LOG_PATHS)
    names = np.array(names)
    N = names.size
    min_maxes = rz.scalar_to_array(N,min_maxes)

    iters = {}
    losses = {}
    if(return_min_max):
        ret_min_maxes = {}
    for (nn,name) in enumerate(names):
        LOSS_LOG_PATH = os.path.join(rootdir,LOSS_LOG_PATHS[nn])
        if(return_min_max):
            (iters[name],losses[name],ret_min_maxes[name]) = parse_loss_log(LOSS_LOG_PATH,iter_norm_factor=iter_norm_factor,\
                return_min_max=True,min_max=min_maxes[nn],mask_max=mask_max)
        else:
            (iters[name],losses[name]) = parse_loss_log(LOSS_LOG_PATH,iter_norm_factor=iter_norm_factor,\
                return_min_max=False,mask_max=mask_max)

    if(return_min_max):
        return (iters,losses,ret_min_maxes)
    else:
        return (iters,losses)

def parse_loss_log(LOSS_LOG_PATH,iter_norm_factor=1000,\
    return_min_max=False,min_max=1,mask_max=True):
    if(os.path.exists(LOSS_LOG_PATH)):
        f = open(LOSS_LOG_PATH,'r')
        cnt = 0
        cur_line = f.readline()
        recs = []
        while(cur_line!=''):
            # print cur_line
            cur_line_split = cur_line.split(',')
            L = len(cur_line_split)-1
            NL = L-2
            cur_rec = np.zeros((L,))
            for (cc,part) in enumerate(cur_line_split[:-1]):
                cur_rec[cc] = float(part.split(' ')[-1])
            recs.append(cur_rec)
            cur_line = f.readline()
        recs = np.array(recs)
        f.close()

        if(mask_max):
            mask = recs[:,1]==np.max(recs[:,1])
        else:
            mask = np.zeros(recs[:,1].size,dtype=bool)+True
        recs = recs[mask]
        recs = np.array(recs)
        iters = recs[:,0]
        Navg = recs[:,1]
        losses = recs[:,2:]
        if(return_min_max):
            if(min_max==0):
                ret_min_max = np.min(losses,axis=0)
            elif(min_max==1):
                ret_min_max = np.max(losses,axis=0)
            # print ret_min_max
            return (iters/iter_norm_factor,losses,ret_min_max)
        else:
            return (iters/iter_norm_factor,losses)
    else:
        if(return_min_max):
            return (np.zeros((1,1)),np.zeros((1,1)),np.zeros((1,1)))
        else:
            return (np.zeros((1,1)),np.zeros((1,1)))

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


class GradientMagnitudeMeterLayer(caffe.Layer):
    ''' Layer which acts as a "meter" to measure gradient magnitude '''
    def setup(self,bottom,top):
        if(len(bottom)==0):
            raise Exception("Layer needs inputs")

        self.cnt = 0 # iteration counter
        self.I = 10 # interval of iterations to keep track
        self.pp = 0
        # self.P = 1 # interval to print gradient magnitudes
        self.P = 10 # interval to print gradient magnitudes
        # self.P = 100 # interval to print gradient magnitudes
        self.h = 0 # index into history
        # self.H = 100 # history size
        self.H = 10 # history size
        self.H_reached = False

        self.L = len(bottom)
        self.Ns = np.zeros((self.L,),dtype=int)
        self.Cs = np.zeros((self.L,),dtype=int)
        self.Xs = np.zeros((self.L,),dtype=int)
        self.Ys = np.zeros((self.L,),dtype=int)

        self.mags = np.zeros((self.L,self.H))

        self.LOG_PATH = './grad_log'

    def reshape(self,bottom,top):
        # print self.L
        for ll in range(self.L):
            self.Ns[ll] = bottom[ll].data.shape[0]
            self.Cs[ll] = bottom[ll].data.shape[1]
            self.Xs[ll] = bottom[ll].data.shape[2]
            self.Ys[ll] = bottom[ll].data.shape[3]
            top[ll].reshape(self.Ns[ll],self.Cs[ll],self.Xs[ll],self.Ys[ll])
        # for ll in range(self.L):

    def forward(self,bottom,top):
        for ll in range(self.L):
            top[ll].data[...] = bottom[ll].data[...] # copy data through

    def backward(self,top,propagate_down,bottom):
        for ll in range(self.L):
            if not propagate_down[ll]:
                continue
            bottom[ll].diff[...] = top[ll].diff[...] # copy diff through

            if(np.mod(self.cnt,self.I)==0): # every Ith iteration, record
                self.mags[ll,self.h] = np.linalg.norm(bottom[ll].diff[...])/self.Ns[ll]

                # if(self.pp==0):
                #     if(self.H_reached==True): # average whole history
                #         print('GradMag %i/%i (%i): %.3f'%(ll,self.L,self.H,np.mean(self.mags[ll,:])))
                #     else: # haven't built whole history yet
                #         print('GradMag %i/%i (%i): %.3f'%(ll,self.L,self.h,np.mean(self.mags[ll,:self.h])))
                #     self.pp = np.mod(self.pp+1,self.P)

        if(np.mod(self.cnt,self.I)==0): # every Ith iteration, record
            if(self.pp==0):
                if(self.H_reached==True): # average whole history
                    tmp_str = '(%i)'%self.H
                    for ll in range(self.L):
                        tmp_str += ' / %.3f'%(np.mean(self.mags[ll,:]))
                else: # haven't built whole history yet
                    tmp_str = '(%i)'%self.h
                    for ll in range(self.L):
                        tmp_str += ' / %.3f'%(np.mean(self.mags[ll,:self.h+1]))
                print_str = 'GradMag: %s'%tmp_str
                print print_str

                self.f = open(self.LOG_PATH,'a')
                self.f.write(print_str)
                self.f.write('\n')
                self.f.close()

            self.pp = np.mod(self.pp+1,self.P)

            if((self.H_reached==False) and (self.h==self.H-1)):
                self.H_reached = True
            self.h = np.mod(self.h+1,self.H)

        self.cnt = self.cnt+1

class ManhattanLossLayer(caffe.Layer):
    ''' Layer which computes L1 loss '''
    def setup(self,bottom,top):
        if(len(bottom)!=2):
            raise Exception("Layer inputs != 2 (len(bottom)!=2)")

        self.N = bottom[0].data.shape[0]

        self.P = np.prod(np.array(bottom[0].data.shape[1:]))
        # self.C = bottom[0].data.shape[1]
        # self.X = bottom[0].data.shape[2]
        # self.Y = bottom[0].data.shape[3]
        # self.P = self.N*self.X*self.Y

    def reshape(self, bottom, top):
        top[0].reshape(1) # single loss value

    def forward(self, bottom, top):
        top[0].data[...] = np.sum(np.abs(bottom[1].data[...]-bottom[0].data[...]))/(self.N*self.P)

    def backward(self, top, propagate_down, bottom):
        sign_diff = np.sign(bottom[1].data[...]-bottom[0].data[...])
        # no back-prop
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            if(i==0):
                bottom[i].diff[...] = -1.*sign_diff/(self.N*self.P)
            else:
                bottom[i].diff[...] = 1.*sign_diff/(self.N*self.P)

class NNEnc2Layer(caffe.Layer):
    ''' Layer which encodes ab map into Q colors
    INPUTS    
        bottom[0]   Nx2xXxY     
    OUTPUTS
        top[0].data     NxQ     
    '''
    def setup(self,bottom,top):
        warnings.filterwarnings("ignore")

        if len(bottom) == 0:
            raise Exception("Layer should have inputs")
        self.NN = 9 # this is hard-coded into the forward
        # self.NN = 1 # this is hard-coded into the forward
        self.sigma = 5.
        self.ENC_DIR = './data/color_bins'
        # self.nnenc = NNEncode(self.NN,self.sigma,km_filepath=os.path.join(self.ENC_DIR,'pts_in_hull.npy'))

        self.pts_in_hull = np.load(os.path.join(self.ENC_DIR,'pts_in_hull.npy'))
        self.prior_probs = np.load(os.path.join(self.ENC_DIR,'prior_probs.npy'))

        self.ENC_DIR = './data/color_bins'
        self.pts_in_hull = np.load(os.path.join(self.ENC_DIR,'pts_in_hull.npy'))
        self.pts_grid = np.load(os.path.join(self.ENC_DIR,'pts_grid.npy'))
        self.prior_probs = np.load(os.path.join(self.ENC_DIR,'prior_probs.npy'))
        self.prior_probs_full = np.load(os.path.join(self.ENC_DIR,'prior_probs_full.npy'))
        self.in_hull = np.load(os.path.join(self.ENC_DIR,'in_hull.npy'))
        self.full_to_hull = np.cumsum(self.in_hull)-1

        self.min_pt = np.min(self.pts_grid)
        self.spacing = np.sort(np.unique(self.pts_grid))
        self.spacing = self.spacing[1] - self.spacing[0]
        self.S = np.sqrt(self.pts_grid.shape[0])

        self.Q = self.pts_in_hull.shape[0]
        self.N = bottom[0].data.shape[0]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]
        self.P = self.N*self.X*self.Y

        self.dists_sq = np.zeros((self.P,self.NN))
        self.inds = np.zeros((self.P,self.NN),dtype='int')

        self.ab_enc_flt = np.zeros((self.P,self.Q))
        self.inds_P = np.arange(0,self.P,dtype='int')[:,rz.na()]

        self.ab_enc_flt_hard = np.zeros((self.P,self.Q))

        if(len(top)==1):
            self.HARD_ENC = False
        else:
            self.HARD_ENC = True

    def reshape(self, bottom, top):
        top[0].reshape(self.N,self.Q,self.X,self.Y) # soft encoding
        if(self.HARD_ENC):
            top[1].reshape(self.N,self.Q,self.X,self.Y) # hard encoding
 
    def forward(self, bottom, top):
        # print 'hello'
        self.ab_enc_flt[...] = 0

        # soft encoding
        ab_val = bottom[0].data[...]
        ab_val_flt = rz.flatten_nd_array(ab_val,axis=1)
        ab_enc_sub = np.round((ab_val-self.min_pt)/self.spacing)
        ab_enc_sub = np.clip(ab_enc_sub,1,self.S-1) # force points into margin

        # ab_enc_sub_flt = rz.flatten_nd_array(ab_enc_sub,axis=1)
        # inds_map = self.full_to_hull[rz.sub2ind2(ab_enc_sub_flt,np.array((self.S,self.S)))]

        t = rz.Timer()
        cnt = 0
        for aa in np.array((0,-1,1)): # hard-coded to find 9-NN
            for bb in np.array((0,-1,1)):
        # for aa in np.array((0,)): # hard-coded to find 1-NN
            # for bb in np.array((0,)):
                tmp = ab_enc_sub.copy()
                tmp[:,0,:,:] = tmp[:,0,:,:]+aa
                tmp[:,1,:,:] = tmp[:,1,:,:]+bb

                ab_enc_sub_flt = rz.flatten_nd_array(tmp,axis=1)
                inds_hull = self.full_to_hull[rz.sub2ind2(ab_enc_sub_flt,np.array((self.S,self.S)))]

                self.dists_sq[:,cnt] = np.sum((ab_val_flt-self.pts_in_hull[inds_hull,:])**2,axis=1)
                self.inds[:,cnt] = inds_hull

                cnt = cnt+1

        # print t.tocStr()
        wts = np.exp(-self.dists_sq/(2*self.sigma**2))
        # print t.tocStr()
        wts = wts/np.sum(wts,axis=1)[:,rz.na()]
        # print t.tocStr()

        self.ab_enc_flt[self.inds_P,self.inds] = wts
        # print t.tocStr()
        top[0].data[...] = rz.unflatten_2d_array(self.ab_enc_flt,ab_val,axis=1)
        # print t.tocStr()

        # hard encoding
        if(self.HARD_ENC):
            self.ab_enc_flt_hard[self.inds_P,self.inds[:,[0]]] = 1
            # print t.tocStr()
            top[1].data[...] = rz.unflatten_2d_array(self.ab_enc_flt_hard,ab_val,axis=1)
            # print t.tocStr()
            self.ab_enc_flt_hard[self.inds_P,self.inds[:,[0]]] = 0
            # print t.tocStr()

    def backward(self, top, propagate_down, bottom):
        # no back-prop
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)

class NNEnc1HotLayer(caffe.Layer):
    ''' Layer which encodes ab map into Q colors
    INPUTS    
        bottom[0]   Nx2xXxY     
    OUTPUTS
        top[0].data     NxQ     
    '''
    def setup(self,bottom,top):
        warnings.filterwarnings("ignore")

        if len(bottom) == 0:
            raise Exception("Layer should have inputs")
        self.NN = 1 # this is hard-coded into the forward
        self.sigma = 5.
        self.ENC_DIR = './data/color_bins'
        # self.nnenc = NNEncode(self.NN,self.sigma,km_filepath=os.path.join(self.ENC_DIR,'pts_in_hull.npy'))

        self.pts_in_hull = np.load(os.path.join(self.ENC_DIR,'pts_in_hull.npy'))
        self.prior_probs = np.load(os.path.join(self.ENC_DIR,'prior_probs.npy'))

        self.ENC_DIR = './data/color_bins'
        self.pts_in_hull = np.load(os.path.join(self.ENC_DIR,'pts_in_hull.npy'))
        self.pts_grid = np.load(os.path.join(self.ENC_DIR,'pts_grid.npy'))
        self.prior_probs = np.load(os.path.join(self.ENC_DIR,'prior_probs.npy'))
        self.prior_probs_full = np.load(os.path.join(self.ENC_DIR,'prior_probs_full.npy'))
        self.in_hull = np.load(os.path.join(self.ENC_DIR,'in_hull.npy'))
        self.full_to_hull = np.cumsum(self.in_hull)-1

        self.min_pt = np.min(self.pts_grid)
        self.spacing = np.sort(np.unique(self.pts_grid))
        self.spacing = self.spacing[1] - self.spacing[0]
        self.S = np.sqrt(self.pts_grid.shape[0])

        self.Q = self.pts_in_hull.shape[0]

    def reshape(self, bottom, top):
        self.N = bottom[0].data.shape[0]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]
        self.P = self.N*self.X*self.Y

        self.dists_sq = np.zeros((self.P,self.NN))
        self.inds = np.zeros((self.P,self.NN),dtype='int')
        self.ab_enc_flt = np.zeros((self.P,self.Q))
        self.inds_P = np.arange(0,self.P,dtype='int')[:,rz.na()]
        self.ab_enc_flt_hard = np.zeros((self.P,self.Q))

        top[0].reshape(self.N,self.Q,self.X,self.Y) # hard encoding
 
    def forward(self, bottom, top):
        self.ab_enc_flt[...] = 0

        # soft encoding
        ab_val = bottom[0].data[...]
        ab_val_flt = rz.flatten_nd_array(ab_val,axis=1)
        ab_enc_sub = np.round((ab_val-self.min_pt)/self.spacing)
        ab_enc_sub = np.clip(ab_enc_sub,1,self.S-1) # force points into margin

        t = rz.Timer()
        cnt = 0
        for aa in np.array((0,)): # hard-coded to find 9-NN
            for bb in np.array((0,)):
                tmp = ab_enc_sub.copy()
                tmp[:,0,:,:] = tmp[:,0,:,:]+aa
                tmp[:,1,:,:] = tmp[:,1,:,:]+bb

                ab_enc_sub_flt = rz.flatten_nd_array(tmp,axis=1)
                inds_hull = self.full_to_hull[rz.sub2ind2(ab_enc_sub_flt,np.array((self.S,self.S)))]

                self.dists_sq[:,cnt] = np.sum((ab_val_flt-self.pts_in_hull[inds_hull,:])**2,axis=1)
                self.inds[:,cnt] = inds_hull

                cnt = cnt+1

        # print t.tocStr()
        wts = np.exp(-self.dists_sq/(2*self.sigma**2))
        # print t.tocStr()
        wts = wts/np.sum(wts,axis=1)[:,rz.na()]
        # print t.tocStr()

        # print np.sum(wts)

        self.ab_enc_flt[self.inds_P,self.inds] = wts
        # print t.tocStr()
        # top[0].data[...] = rz.unflatten_2d_array(self.ab_enc_flt,ab_val,axis=1)
        # print t.tocStr()

        # hard encoding
        # if(self.HARD_ENC):
        self.ab_enc_flt_hard[self.inds_P,self.inds[:,[0]]] = 1
            # print t.tocStr()
        top[0].data[...] = rz.unflatten_2d_array(self.ab_enc_flt_hard, ab_val, axis=1)
            # print t.tocStr()
            # self.ab_enc_flt_hard[self.inds_P,self.inds[:,[0]]] = 0
            # print t.tocStr()

    def backward(self, top, propagate_down, bottom):
        # no back-prop
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)

# ************************
# ***** CAFFE LAYERS *****
# ************************
class BGR2HSVLayer(caffe.Layer):
    ''' Layer converts BGR to HSV
    INPUTS    
        bottom[0]   Nx3xXxY     
    OUTPUTS
        top[0].data     Nx3xXxY     
    '''
    def setup(self,bottom, top):
        warnings.filterwarnings("ignore")

        if(len(bottom)!=1):
            raise Exception("Layer should a single input")
        if(bottom[0].data.shape[1]!=3):
            raise Exception("Input should be 3-channel BGR image")

        self.N = bottom[0].data.shape[0]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]

    def reshape(self, bottom, top):
        top[0].reshape(self.N,3,self.X,self.Y)
 
    def forward(self, bottom, top):
        for nn in range(self.N):
            top[0].data[nn,:,:,:] = color.rgb2hsv(bottom[0].data[nn,::-1,:,:].astype('uint8').transpose((1,2,0))).transpose((2,0,1))

    def backward(self, top, propagate_down, bottom):
        # no back-prop
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            # bottom[i].diff[...] = np.zeros_like(bottom[i].data)

class BGR2LabLayer(caffe.Layer):
    ''' Layer converts BGR to Lab
    INPUTS    
        bottom[0]   Nx3xXxY     
    OUTPUTS
        top[0].data     Nx3xXxY     
    '''
    def setup(self,bottom, top):
        warnings.filterwarnings("ignore")

        if(len(bottom)!=1):
            raise Exception("Layer should a single input")
        if(bottom[0].data.shape[1]!=3):
            raise Exception("Input should be 3-channel BGR image")

        self.N = bottom[0].data.shape[0]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]

    def reshape(self, bottom, top):
        top[0].reshape(self.N,3,self.X,self.Y)
 
    def forward(self, bottom, top):
        top[0].data[...] = color.rgb2lab(bottom[0].data[:,::-1,:,:].astype('uint8').transpose((2,3,0,1))).transpose((2,3,0,1))

    def backward(self, top, propagate_down, bottom):
        # no back-prop
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            # bottom[i].diff[...] = np.zeros_like(bottom[i].data)

class EncLayer(caffe.Layer):
    ''' Layer which does hard quantization into bins
    INPUTS    
        bottom[0]   Nx1xXxY     
    OUTPUTS
        top[0].data     NxQ     
    '''
    def setup(self,bottom, top):
        warnings.filterwarnings("ignore")

        if len(bottom) == 0:
            raise Exception("Layer should have inputs")

        self.param_str_split = self.param_str.split(' ')
        self.min = float(self.param_str_split[0])
        self.max = float(self.param_str_split[1])
        self.inc = float(self.param_str_split[2])

        self.N = bottom[0].data.shape[0]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]

    def reshape(self, bottom, top):
        top[0].reshape(self.N,1,self.X,self.Y)
 
    def forward(self, bottom, top):
        top[0].data[...] = (bottom[0].data[...]-self.min)/self.inc

    def backward(self, top, propagate_down, bottom):
        # no back-prop
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)

class NNEncLayer(caffe.Layer):
    ''' Layer which encodes ab map into Q colors
    INPUTS    
        bottom[0]   Nx2xXxY     
    OUTPUTS
        top[0].data     NxQ     
    '''
    def setup(self,bottom,top):
        warnings.filterwarnings("ignore")

        if len(bottom) == 0:
            raise Exception("Layer should have inputs")
        self.NN = 10.
        self.sigma = 5.
        self.ENC_DIR = './data/color_bins'
        self.nnenc = NNEncode(self.NN,self.sigma,km_filepath=os.path.join(self.ENC_DIR,'pts_in_hull.npy'))

        self.HARD_FLAG = False
        if(len(top)==2):
            self.nnenc2 = NNEncode(1,self.sigma,km_filepath=os.path.join(self.ENC_DIR,'pts_in_hull.npy'))
            self.HARD_FLAG = True

        self.N = bottom[0].data.shape[0]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]
        self.Q = self.nnenc.K

    def reshape(self, bottom, top):
        self.N = bottom[0].data.shape[0]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]
        self.Q = self.nnenc.K
        top[0].reshape(self.N,self.Q,self.X,self.Y)
        if(self.HARD_FLAG):
            top[1].reshape(self.N,self.Q,self.X,self.Y)
 
    def forward(self, bottom, top):
        # print bottom[0].data.shape
        # top[0].data[...] = self.nnenc.encode_points_mtx_nd(bottom[0].data[...],axis=1)
        if(self.HARD_FLAG):
            top[1].data[...] = self.nnenc2.encode_points_mtx_nd(bottom[0].data[...],axis=1)

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
    ''' Layer outputs a mask based on if the image is grayscale or not
    INPUTS    
        bottom[0]       Nx2xXxY     ab values
    OUTPUTS
        top[0].data     Nx1xXxY     1 if image is NOT grayscale
                                    0 if image is grayscale
    '''
    def setup(self,bottom, top):
        if len(bottom) == 0:
            raise Exception("Layer should have inputs")

        self.thresh = 5 # threshold on ab value
        self.N = bottom[0].data.shape[0]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]

    def reshape(self, bottom, top):
        top[0].reshape(self.N,1,self.X,self.Y)
 
    def forward(self, bottom, top):
        # if an image has any (a,b) value which exceeds threshold, output 1
        top[0].data[...] = (np.sum(np.sum(np.sum(np.abs(bottom[0].data) > self.thresh,axis=1),axis=1),axis=1) > 0)[:,na(),na(),na()]

    def backward(self, top, propagate_down, bottom):
        # no back-prop
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)

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

# ***************************
# ***** SUPPORT CLASSES *****
# ***************************
class PriorFactor():
    ''' Class handles prior factor '''
    # def __init__(self,alpha,gamma=0,verbose=True,priorFile='/home/eecs/rich.zhang/src/projects/cross_domain/save/ab_grid_10/prior_probs.npy',genc=-1):
    def __init__(self,alpha,gamma=0,verbose=True,priorFile=''):
        # INPUTS
        #   alpha           integer     prior correction factor, 0 to ignore prior, 1 to divide by prior, alpha to divide by prior^alpha power
        #   gamma           integer     percentage to mix in prior probability
        #   priorFile       file        file which contains prior probabilities across classes    

        # settings
        self.alpha = alpha
        self.gamma = gamma
        self.verbose = verbose

        # empirical prior probability
        self.prior_probs = np.load(priorFile)

        # define uniform probability
        self.uni_probs = np.zeros_like(self.prior_probs)
        self.uni_probs[self.prior_probs!=0] = 1.
        self.uni_probs = self.uni_probs/np.sum(self.uni_probs)

        # convex combination of empirical prior and uniform distribution       
        self.prior_mix = (1-self.gamma)*self.prior_probs + self.gamma*self.uni_probs

        # set prior factor
        self.prior_factor = self.prior_mix**-self.alpha
        self.prior_factor = self.prior_factor/np.sum(self.prior_probs*self.prior_factor) # re-normalize

        # implied empirical prior
        self.implied_prior = self.prior_probs*self.prior_factor
        self.implied_prior = self.implied_prior/np.sum(self.implied_prior) # re-normalize

        # add this to the softmax score
        # self.softmax_correction = np.log(self.prior_probs/self.implied_prior * (1-self.implied_prior)/(1-self.prior_probs))

        if(self.verbose):
            self.print_correction_stats()

        # if(not check_value(genc,-1)):
            # self.expand_grid(genc)

    # def expand_grid(self,genc):
    #     self.prior_probs_full_grid = genc.enc_full_grid_mtx_nd(self.prior_probs,axis=0,returnGrid=True)
    #     self.uni_probs_full_grid = genc.enc_full_grid_mtx_nd(self.uni_probs,axis=0,returnGrid=True)
    #     self.prior_mix_full_grid = genc.enc_full_grid_mtx_nd(self.prior_mix,axis=0,returnGrid=True)
    #     self.prior_factor_full_grid = genc.enc_full_grid_mtx_nd(self.prior_factor,axis=0,returnGrid=True)
    #     self.implied_prior_full_grid = genc.enc_full_grid_mtx_nd(self.implied_prior,axis=0,returnGrid=True)
    #     self.softmax_correction_full_grid = genc.enc_full_grid_mtx_nd(self.softmax_correction,axis=0,returnGrid=True)

    def print_correction_stats(self):
        print 'Prior factor correction:'
        print '  (alpha,gamma) = (%.2f, %.2f)'%(self.alpha,self.gamma)
        print '  (min,max,mean,med,exp) = (%.2f, %.2f, %.2f, %.2f, %.2f)'%(np.min(self.prior_factor),np.max(self.prior_factor),np.mean(self.prior_factor),np.median(self.prior_factor),np.sum(self.prior_factor*self.prior_probs))

    def forward(self,data_ab_quant,axis=1):
        # data_ab_quant = net.blobs['data_ab_quant_map_233'].data[...]
        data_ab_maxind = np.argmax(data_ab_quant,axis=axis)
        corr_factor = self.prior_factor[data_ab_maxind]
        if(axis==0):
            return corr_factor[na(),:]
        elif(axis==1):
            return corr_factor[:,na(),:]
        elif(axis==2):
            return corr_factor[:,:,na(),:]
        elif(axis==3):
            return corr_factor[:,:,:,na()]

class NNEncode():
    ''' Encode points using NN search and Gaussian kernel '''
    def __init__(self,NN,sigma,km_filepath='',cc=-1):
        if(check_value(cc,-1)):
            self.cc = np.load(km_filepath)
        else:
            self.cc = cc
        self.K = self.cc.shape[0]
        # self.NN = NN
        self.NN = int(NN)
        self.sigma = sigma
        self.nbrs = nn.NearestNeighbors(n_neighbors=NN, algorithm='ball_tree').fit(self.cc)

        self.alreadyUsed = False

    def encode_points_mtx_nd(self,pts_nd,axis=1,returnSparse=False,sameBlock=True):
        t = rz.Timer();
        pts_flt = flatten_nd_array(pts_nd,axis=axis)
        P = pts_flt.shape[0]
        if(sameBlock and self.alreadyUsed):
            self.pts_enc_flt[...] = 0 # already pre-allocated
        else:
            self.alreadyUsed = True
            self.pts_enc_flt = np.zeros((P,self.K))
            self.p_inds = np.arange(0,P,dtype='int')[:,na()]

        P = pts_flt.shape[0]

        (dists,inds) = self.nbrs.kneighbors(pts_flt)

        wts = np.exp(-dists**2/(2*self.sigma**2))
        wts = wts/np.sum(wts,axis=1)[:,na()]

        self.pts_enc_flt[self.p_inds,inds] = wts
        pts_enc_nd = unflatten_2d_array(self.pts_enc_flt,pts_nd,axis=axis)

        return pts_enc_nd

    def decode_points_mtx_nd(self,pts_enc_nd,axis=1):
        pts_enc_flt = flatten_nd_array(pts_enc_nd,axis=axis)
        pts_dec_flt = np.dot(pts_enc_flt,self.cc)
        pts_dec_nd = unflatten_2d_array(pts_dec_flt,pts_enc_nd,axis=axis)
        return pts_dec_nd

    def decode_1hot_mtx_nd(self,pts_enc_nd,axis=1,returnEncode=False):
        pts_1hot_nd = nd_argmax_1hot(pts_enc_nd,axis=axis)
        pts_dec_nd = self.decode_points_mtx_nd(pts_1hot_nd,axis=axis)
        if(returnEncode):
            return (pts_dec_nd,pts_1hot_nd)
        else:
            return pts_dec_nd

# *****************************
# ***** Utility functions *****
# *****************************
def check_value(inds, val):
    ''' Check to see if an array is a single element equaling a particular value
    for pre-processing inputs in a function '''
    if(np.array(inds).size==1):
        if(inds==val):
            return True
    return False

def na(): # shorthand for new axis
    return np.newaxis

def flatten_nd_array(pts_nd,axis=1):
    ''' Flatten an nd array into a 2d array with a certain axis
    INPUTS
        pts_nd         N0xN1x...xNd array
        axis         integer
    OUTPUTS
        pts_flt     prod(N \ N_axis) x N_axis array     '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
    NPTS = np.prod(SHP[nax])
    axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
    pts_flt = pts_nd.transpose((axorder))
    pts_flt = pts_flt.reshape(NPTS,SHP[axis])
    return pts_flt

def unflatten_2d_array(pts_flt,pts_nd,axis=1,squeeze=False):
    ''' Unflatten a 2d array with a certain axis
    INPUTS
        pts_flt     prod(N \ N_axis) x M array
        pts_nd         N0xN1x...xNd array
        axis         integer
        squeeze     bool     if true, M=1, squeeze it out
    OUTPUTS
        pts_out     N0xN1x...xNd array        '''
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