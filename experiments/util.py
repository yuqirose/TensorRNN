import sys

import cudamat as cm
from cudamat import cudamat_conv_gemm as cc
from cudamat import gpu_lock2 as gpu_lock
import h5py

import os
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from time import sleep
import pdb
import datetime
import time
import config_pb2
from google.protobuf import text_format
from random import randint

# Parameter is preety much a weight (consisting of weights and derivatives of weights)
class Param(object):
    def __init__(self, w, config=None):
        if type(w) == np.ndarray:
            self.w_ = cm.CUDAMatrix(w)
        elif type(w) == tuple:
            self.w_ = cm.empty(w)
        else:
            self.w_ = w
        self.dw_ = cm.empty_like(self.w_)
        self.dw_history_ = cm.empty_like(self.w_)
        self.dw_history_sqrt_ = cm.empty_like(self.w_)
        self.dw_sqr_ = cm.empty_like(self.w_)
        self.dw_history_.assign(0)
        self.dw_.assign(0)
        self.t_ = 0

        if config is None:
            pass
        elif config.init_type == config_pb2.Param.CONSTANT:
            self.w_.assign(config.scale)
        elif config.init_type == config_pb2.Param.GAUSSIAN:
            self.w_.fill_with_randn()
            self.w_.mult(config.scale)
        elif config.init_type == config_pb2.Param.UNIFORM:
            self.w_.fill_with_rand()
            self.w_.subtract(0.5)
            self.w_.mult(2 * config.scale)
        elif config.init_type == config_pb2.Param.PRETRAINED:
            f = h5py.File(config.file_name)
            mat = f[config.dataset_name].value
            if len(mat.shape) == 1:
                mat = mat.reshape(1, -1)
            assert self.w_.shape == mat.shape
            self.w_.overwrite(mat)
            f.close()
        else:
            raise Exception('Unknown parameter initialization.')

        self.eps_ = config.epsilon
        self.momentum_ = config.momentum
        self.l2_decay_ = config.l2_decay
        self.rmsprop_decay_ = 0.9
        self.gradient_clip_ = config.gradient_clip
        self.eps_decay_factor = config.eps_decay_factor
        self.eps_decay_after = config.eps_decay_after

    def __repr__(self):
        return self.w_.asarray().__repr__()
    
    def __str__(self):
        return self.w_.asarray().__str__()

    def Load(self, f, name):
        if name in f.keys():
            self.w_.overwrite(f[name].value)
            self.dw_history_.overwrite(f['%s_grad' % name].value)
            self.t_ = f.attrs.get('%s_t' % name, 0)
        else:
            print("%s not found." % name)

    def Save(self, f, name):
        w_dset = f.create_dataset(name, self.w_.shape, dtype=np.float32)
        w_dset[:, :] = self.w_.asarray()
        w_dset = f.create_dataset('%s_grad' % name, self.dw_history_.shape, dtype=np.float32)
        w_dset[:, :] = self.dw_history_.asarray()
        f.attrs.__setitem__('%s_t' % name, self.t_)

    def GetW(self):
        return self.w_
    
    def GetdW(self):
        return self.dw_

    def Update(self):
        if self.eps_decay_after > 0:
            eps = self.eps_ * np.power(self.eps_decay_factor, self.t_ / self.eps_decay_after)
        else:
            eps = self.eps_

        """
        # RMSPROP
        self.dw_history_.mult(self.rmsprop_decay_)
        self.dw_.mult(self.dw_, target=self.dw_sqr_)
        self.dw_history_.add_mult(self.dw_sqr_, mult=(1-self.rmsprop_decay_))
        self.dw_history_.add(1e-07)
        cm.sqrt(self.dw_history_, target=self.dw_history_sqrt_)

        self.dw_.mult(-self.eps_)
        self.dw_.divide(self.dw_history_sqrt_)

        if self.gradient_clip_ > 0:
            self.dw_.upper_bound_mod(self.gradient_clip_)
        self.w_.add(self.dw_)
        """

        self.dw_history_.mult(self.momentum_)
        self.dw_.add_mult(self.w_, mult=self.l2_decay_)
        self.dw_history_.add_mult(self.dw_, -self.eps_)
        if self.gradient_clip_ > 0:
            self.dw_history_.upper_bound_mod(self.gradient_clip_)
        self.w_.add(self.dw_history_)

        self.t_ += 1

def ReadDataProto(fname):
    data_pb = config_pb2.Data()
    with open(fname, 'r') as pbtxt:
        text_format.Merge(pbtxt.read(), data_pb)
    return data_pb

def ReadModelProto(fname):
    data_pb = config_pb2.Model()
    with open(fname, 'r') as pbtxt:
        text_format.Merge(pbtxt.read(), data_pb)
    return data_pb

def WritePbtxt(proto, fname):
    with open(fname, 'w') as f:
        text_format.PrintMessage(proto, f)

def LockGPU(max_retries=10, board=-1):

    # Assuming you already got GPU lock
    cm.cuda_set_device(board)
    cm.cublas_init()
    return board

def FreeGPU(board):
    cm.cublas_shutdown()

    

class BouncingMNISTDataHandler(object):
    """Data Handler that creates Bouncing MNIST dataset on the fly."""
    def __init__(self, config):
        self.seq_length_ = config.num_frames
        self.batch_size_ = config.batch_size
        self.image_size_ = config.image_size
        self.num_digits_ = config.num_digits
        self.step_length_ = config.step_length
        self.dataset_size_ = 10000  # The dataset is really infinite. This is just for validation.
        self.digit_size_ = 28
        self.frame_size_ = self.image_size_ ** 2

        try:
            f = h5py.File('/Users/roseyu/Documents/Python/TensorRNN/datasets/mnist.h5')
        except:
            print('Please set the correct path to MNIST dataset')
            sys.exit()

        self.data_ = f['train'].value.reshape(-1, 28, 28)
        f.close()
        self.indices_ = np.arange(self.data_.shape[0])
        self.row_ = 0
        np.random.shuffle(self.indices_)

    def GetBatchSize(self):
        return self.batch_size_

    def GetDims(self):
        return self.frame_size_

    def GetDatasetSize(self):
        return self.dataset_size_

    def GetSeqLength(self):
        return self.seq_length_

    def Reset(self):
        pass

    def GetRandomTrajectory(self, batch_size):
        length = self.seq_length_
        canvas_size = self.image_size_ - self.digit_size_
        
        # Initial position uniform random inside the box.
        y = np.random.rand(batch_size)
        x = np.random.rand(batch_size)

        # Choose a random velocity.
        theta = np.random.rand(batch_size) * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros((length, batch_size))
        start_x = np.zeros((length, batch_size))
        for i in xrange(length):
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            # Bounce off edges.
            for j in xrange(batch_size):
                if x[j] <= 0:
                    x[j] = 0
                    v_x[j] = -v_x[j]
                if x[j] >= 1.0:
                    x[j] = 1.0
                    v_x[j] = -v_x[j]
                if y[j] <= 0:
                    y[j] = 0
                    v_y[j] = -v_y[j]
                if y[j] >= 1.0:
                    y[j] = 1.0
                    v_y[j] = -v_y[j]
            start_y[i, :] = y
            start_x[i, :] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def Overlap(self, a, b):
        """ Put b on top of a."""
        return np.maximum(a, b)
        #return b

    def GetBatch(self, verbose=False):
        start_y, start_x = self.GetRandomTrajectory(self.batch_size_ * self.num_digits_)
        
        # minibatch data
        data = np.zeros((self.batch_size_, self.seq_length_, self.image_size_, self.image_size_), dtype=np.float32)
        
        for j in xrange(self.batch_size_):
            for n in xrange(self.num_digits_):
             
                # get random digit from dataset
                ind = self.indices_[self.row_]
                self.row_ += 1
                if self.row_ == self.data_.shape[0]:
                    self.row_ = 0
                    np.random.shuffle(self.indices_)
                digit_image = self.data_[ind, :, :]
                
                # generate video
                for i in xrange(self.seq_length_):
                    top    = start_y[i, j * self.num_digits_ + n]
                    left   = start_x[i, j * self.num_digits_ + n]
                    bottom = top  + self.digit_size_
                    right  = left + self.digit_size_
                    data[j, i, top:bottom, left:right] = self.Overlap(data[j, i, top:bottom, left:right], digit_image)
        
        return data.reshape(self.batch_size_, -1), None

    def DisplayData(self, data, rec=None, fut=None, fig=1, case_id=0, output_file=None):
        output_file1 = None
        output_file2 = None
        
        if output_file is not None:
            name, ext = os.path.splitext(output_file)
            output_file1 = '%s_original%s' % (name, ext)
            output_file2 = '%s_recon%s' % (name, ext)
        
        # get data
        data = data[case_id, :].reshape(-1, self.image_size_, self.image_size_)
        # get reconstruction and future sequences if exist
        if rec is not None:
            rec = rec[case_id, :].reshape(-1, self.image_size_, self.image_size_)
            enc_seq_length = rec.shape[0]
        if fut is not None:
            fut = fut[case_id, :].reshape(-1, self.image_size_, self.image_size_)
            if rec is None:
                enc_seq_length = self.seq_length_ - fut.shape[0]
            else:
                assert enc_seq_length == self.seq_length_ - fut.shape[0]
        
        num_rows = 1
        # create figure for original sequence
        plt.figure(2*fig, figsize=(20, 1))
        plt.clf()
        for i in xrange(self.seq_length_):
            plt.subplot(num_rows, self.seq_length_, i+1)
            plt.imshow(data[i, :, :], cmap=plt.cm.gray, interpolation="nearest")
            plt.axis('off')
        plt.draw()
        if output_file1 is not None:
            print(output_file1)
            plt.savefig(output_file1, bbox_inches='tight')

        # create figure for reconstuction and future sequences
        plt.figure(2*fig+1, figsize=(20, 1))
        plt.clf()
        for i in xrange(self.seq_length_):
            if rec is not None and i < enc_seq_length:
                plt.subplot(num_rows, self.seq_length_, i + 1)
                plt.imshow(rec[rec.shape[0] - i - 1, :, :], cmap=plt.cm.gray, interpolation="nearest")
            if fut is not None and i >= enc_seq_length:
                plt.subplot(num_rows, self.seq_length_, i + 1)
                plt.imshow(fut[i - enc_seq_length, :, :], cmap=plt.cm.gray, interpolation="nearest")
            plt.axis('off')
        plt.draw()
        if output_file2 is not None:
            print(output_file2)
            plt.savefig(output_file2, bbox_inches='tight')
        else:
            plt.pause(0.1)


