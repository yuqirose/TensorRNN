import argparse
import sys
import time
import torch
import torch.nn as nn 
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from seq_model import *
from seq_io import *


def train(args):
    # set ramdom seed to 0
    np.random.seed(0)
    torch.manual_seed(0)

    # load data and make training set
    train_seq, valid_seq, test_seq = seq_raw_data(args.data)

    eval_batch_size = 10
    train_data = seq_to_batch(train_seq, args.batch_size)
    val_data = seq_to_batch(valid_seq, eval_batch_size)
    test_data = seq_to_batch(test_seq, eval_batch_size)

    # build the model
    ndim = train_data.size(2)
    model = Seq_LSTM(ndim, args.nhid, args.nlayers)

    model.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(model.parameters(), lr= 0.01)

    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        hidden = tuple(h.detach() for h in hidden)
        model.zero_grad()
        input, targets = get_batch(train_data, i, args.bptt)
        print('STEP: ', i)
        def closure(): # closure allow recompute model, for L-BFGS
            optimizer.zero_grad()
            out = model(input, hidden)
            loss = criterion(out, target)
            print('loss:', loss.data.numpy()[0])
            loss.backward()
            return loss
        optimizer.step(closure)

        # begin to predict
        future = 1000 #forecast horizon
        pred = seq(input[:3], future = future)
        y = pred.data.numpy()

def eval(data_source, args):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)

    
def main():

    parser = argparse.ArgumentParser(description="parser for tensor-rnn")
    parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
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
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str,  default='model.pt',
                        help='path to save the final model')
    args = parser.parse_args()


    if not torch.cuda.is_available():
        print("WARNING: cuda is not available, try running on CPU")

    #begin to train
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train(args)
            val_loss = eval(val_data)
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
 