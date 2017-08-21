import torch
import torch.nn as nn 
from torch.autograd import Variable


class Seq_LSTM(nn.Module):
    def __init__(self, ninp, nhid, nlayers):
        super(Seq_LSTM, self).__init__()
        self.rnn = nn.LSTM(ninp, nhid, nlayers)
        self.decoder = nn.Linear(nhid, ninp)

        self.nhid = nhid
        self.nlayers = nlayers

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        hidden = (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))

        return hidden

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
