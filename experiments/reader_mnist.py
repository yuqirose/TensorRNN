from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import random_seed
import h5py


# start of sequences
SOS = 0

class MNISTDataSet(object):
    def __init__(self,
                     data,
                     input_steps,
                     output_steps,
                     seed=None):
        """Construct a DataSet.
        Seed arg provides for convenient deterministic testing.
        """
        self.data_ = data # N x H x W
        self.num_examples_ = data.shape[0]
        self.input_steps_ = input_steps
        self.output_steps_ = output_steps
        self.seq_length_ = input_steps + output_steps
        self.image_size_ = data.shape[-1]
        self.num_digits_ = 2
        self.digit_size_ = 28
        self.step_length_ = 12 #trajectory speed
        self.frame_size_ = self.image_size_ ** 2
        self._epochs_completed = 0
        self._index_in_epoch = 0

        self.indices_ = np.arange(self.num_examples_)

        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        np.random.seed(seed1 if seed is None else seed2)
        
        if output_steps is None:
            output_steps=  self.seq_length_-input_steps

        # enc_inps = data[:,:input_steps, :] 
        # dec_inps = np.insert(data[:,input_steps:input_steps+output_steps-1,:], 0, SOS, axis=1)
        # #dec_outs = np.insert(data[:,input_steps:input_steps+output_steps,:], output_steps, EOS, axis=1)
        # dec_outs = data[:,input_steps:input_steps+output_steps,:] 

        # assert enc_inps.shape[0] == dec_outs.shape[0], (
        #         'inps.shape: %s outs.shape: %s' % (inps.shape, outs.shape))

    @property
    def enc_inps(self):
        return self.enc_inps_
    @property
    def dec_inps(self):
        return self.dec_inps_
    @property
    def dec_outs(self):
        return self.dec_outs_

    @property
    def num_examples(self):
        return self.num_examples_

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def gen_trajectory(self, batch_size):
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

    def gen_video(self, ind, data_size):
        start_y, start_x = self.gen_trajectory(data_size * self.num_digits_)
        
        # minibatch data
        data = np.zeros((data_size, self.seq_length_, self.image_size_, self.image_size_), dtype=np.float32)
        
        for j in xrange(data_size):
            for n in xrange(self.num_digits_):
             
                # get a digit from dataset
                digit_image = self.data_[ind, :]   
                # generate video
                for i in xrange(self.seq_length_):
                    top    = start_y[i, j * self.num_digits_ + n]
                    left   = start_x[i, j * self.num_digits_ + n]
                    bottom = top  + self.digit_size_
                    right  = left + self.digit_size_
                    data[j, i, top:bottom, left:right] = self.Overlap(data[j, i, top:bottom, left:right], digit_image)
        
        return data

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            np.random.shuffle(self.indices_)
            batch_video = self.gen_video(self.indices_[start], batch_size)
            # self._enc_inps = self.enc_inps[perm0]
            # self._dec_inps = self.dec_inps[perm0]
            # self._dec_outs = self.dec_outs[perm0]

        # Go to the next epoch
        if start + batch_size >= self.num_examples_:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self.num_examples_ - start
            batch_video_rest_part = self.gen_video(self.indices_[start], rest_num_examples)
            # enc_inps_rest_part = self._enc_inps[start:self._num_examples]
            # dec_inps_rest_part = self._dec_inps[start:self._num_examples]
            # dec_outs_rest_part = self._dec_outs[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                np.random.shuffle(self.indices_)
                batch_video = self.gen_video(self.indices_[start], batch_size) 
                # self._enc_inps = self.enc_inps[perm]
                # self._dec_inps = self.dec_inps[perm]
                # self._dec_outs = self.dec_outs[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            new_num_examples= self._index_in_epoch
            batch_video_new_part = self.gen_video(self.indices_[start], new_num_examples)
            batch_video = np.concatenate((batch_video_rest_part, batch_video_new_part), axis=0)

            # enc_inps_new_part = self._enc_inps[start:end]
            # dec_inps_new_part = self._dec_inps[start:end]
            # dec_outs_new_part = self._dec_outs[start:end]

            # return np.concatenate((enc_inps_rest_part, enc_inps_new_part), axis=0), \
            #        np.concatenate((dec_inps_rest_part, dec_inps_new_part), axis=0), \
            #        np.concatenate((dec_outs_rest_part, dec_outs_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            batch_video = self.gen_video(self.indices_[start], batch_size)

        self.enc_inps_ = batch_video[:,:self.input_steps_,:].reshape(-1, self.input_steps_, self.frame_size_)
        self.dec_inps_ = np.insert(batch_video[:, self.input_steps_:self.seq_length_-1,:], 0, SOS, axis=1).reshape(-1, self.output_steps_, self.frame_size_) 
        self.dec_outs_ = batch_video[:, self.input_steps_:self.seq_length_,:].reshape(-1, self.output_steps_, self.frame_size_) 

        return self.enc_inps_, self.dec_inps_, self.dec_outs_

    def DisplayData(self, data, rec=None, fut=None, fig=1, case_id=0, output_file=None):
        output_file1 = None
        output_file2 = None
        
        if output_file is not None:
            name, ext = os.path.splitext(output_file)
            output_file1 = '%s_original%s' % (name, ext)
            output_file2 = '%s_recon%s' % (name, ext)
        
        # get data: T x H x W
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
        

def read_data_sets(data_path, input_steps,
                                output_steps,
                                val_size = 0.1, 
                                test_size = 0.1, 
                                seed=None):
    print("loading time series ...")
    image_size = 28
    f = h5py.File(data_path)
    data = f['train'].value.reshape(-1, image_size, image_size)

    # Expand the dimension if univariate time series
    if (np.ndim(data)==1):
            data = np.expand_dims(data, axis=1)
    # Normalize the data
    # print("normalize to (0-1)")
    # data, _ = normalize_columns(data)


    ntest = int(round(len(data) * (1.0 - test_size)))
    nval = int(round(len(data[:ntest]) * (1.0 - val_size)))

    train_data, valid_data, test_data = data[:nval, ], data[nval:ntest, ], data[ntest:,]

    train_options = dict(input_steps=input_steps, output_steps=output_steps, seed=seed)
    
    train = MNISTDataSet(train_data, **train_options)
    valid = MNISTDataSet(valid_data, **train_options)
    test = MNISTDataSet(test_data, **train_options)
   

    stats ={}
    stats['num_examples'] = data.shape[0]
    stats['num_input'] = image_size**2

    return base.Datasets(train=train, validation=valid, test=test), stats
