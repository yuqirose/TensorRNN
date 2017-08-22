import argparse
import sys
import time
import torch
import math
import torch.nn as nn 
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from seq_model import *
from seq_io import *


def train(model,train_data,criterion, args):
    model.train()
    total_loss = 0.0   
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    print('train data size', train_data.size(0) - 1)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        input, target = get_batch(train_data, i, args.bptt)
        
        hidden = tuple(h.detach() for h in hidden)
        model.zero_grad()
        # print('input shape', input.data.size(), 'target shape', target.data.size())
        # print('input tensor:', print(input.data.numpy()[:,0,0]), 'target tensor:', print(target.data.numpy()[:,0,0]))
        print('STEP: ', batch)
        out, hidden = model(input, hidden)
        # print('target tensor:', print(target.data.numpy()[:5,0,0]),'output tensor:',print(out.data.numpy()[:5,0,0]))
        loss = criterion(out, target)
        loss.backward()

        total_loss += loss.data
    print('loss', total_loss.numpy()[0])
    

 


def eval(model, data_source, criterion,args):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(args.eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args.bptt, evaluation=True)
        output, hidden = model(data, hidden)

        total_loss += len(data) * criterion(output, targets).data
        hidden = tuple(h.detach() for h in hidden)
    return total_loss[0] / len(data_source)

    
def main():

    parser = argparse.ArgumentParser(description="parser for tensor-rnn")
    parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                        help='eval batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str,  default='model.pt',
                        help='path to save the final model')
    args = parser.parse_args()


    if not torch.cuda.is_available():
        print("WARNING: cuda is not available, try running on CPU")


    lr = args.lr
    best_val_loss = None
    # load data and make training set
    train_seq, valid_seq, test_seq = seq_raw_data(args.data)

    train_data = seq_to_batch(train_seq, args.batch_size)
    val_data = seq_to_batch(valid_seq, args.eval_batch_size)
    test_data = seq_to_batch(test_seq, args.eval_batch_size)

    # set ramdom seed to 0
    np.random.seed(0)
    torch.manual_seed(0)

    # build the model
    ndim = train_data.size(2)
    model = Seq_LSTM(ndim, args.nhid, args.nlayers)

    # model.double()
    criterion = nn.MSELoss()

    #begin to train
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train(model, train_data, criterion, args)
            val_loss = eval(model, val_data, criterion,args)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')



if __name__ == '__main__':
    main()
 