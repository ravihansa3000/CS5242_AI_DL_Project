import torch
from torch import nn
import torch.nn.functional as F
import random
from torch.autograd import Variable


class S2VTModel(nn.Module):
    def __init__(self, vocab_size=117, dim_hidden=500, dim_word=500, max_len=3, dim_vid=500, sos_id=1000,
                 n_layers=1, rnn_cell='lstm', rnn_dropout_p=0.2):
        super(S2VTModel, self).__init__()
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        # features of video frames are embedded to a 500 dimensional space
        self.dim_vid = dim_vid

        # object vocab: 35 + relationships vocab: 82
        self.dim_output = vocab_size
        self.dim_hidden = dim_hidden

        # words are transformed to 500 feature dimension
        self.dim_word = dim_word

        # only 3 words are predicted <object1, relationship, object2>
        self.max_length = max_len

        # start-of-sentence id
        self.sos_id = sos_id

        # word embeddings lookup table with
        self.embedding = nn.Embedding(self.dim_output, self.dim_word)

        self.rnn1 = self.rnn_cell(self.dim_vid, self.dim_hidden, n_layers, batch_first=True, dropout=rnn_dropout_p)
        self.rnn2 = self.rnn_cell(self.dim_hidden + self.dim_word, self.dim_hidden, n_layers,
                                  batch_first=True, dropout=rnn_dropout_p)

        self.out = nn.Linear(self.dim_hidden, self.dim_output)

    def forward(self, vid_feats: torch.Tensor, target_variable=None, mode='train', opt={}):
        """
        :param vid_feats: Tensor containing video features

        :param target_variable: array containing annotations.
            Each row corresponds to a set of training annotations; (SOS, object1, relationship, object2, EOS)

        :param mode: 'train' or 'test'

        :param opt: not used

        :return:
        """

        batch_size, n_frames, _ = vid_feats.shape

        # https://github.com/pytorch/pytorch/issues/3920
        # paddings to be used for the 2nd layer
        padding_words = Variable(torch.empty(batch_size, n_frames, self.dim_word, dtype=vid_feats.dtype)).zero_()

        # paddings to be used for the 1st layer
        padding_frames = Variable(torch.empty(batch_size, 1, self.dim_vid, dtype=vid_feats.dtype)).zero_()

        # hidden and cell states of 2 LSTM layers
        state1 = None
        state2 = None

        # feed the video features into the first layer of RNN
        # only 30 steps will be performed since n_frames is always 30 (based on train/test data)
        output1, state1 = self.rnn1(vid_feats, state1)

        # concatenate paddings (for the 2nd layer) with output from the 1st layer, use the 3rd dimension to concat
        input2 = torch.cat((output1, padding_words), dim=2)

        # feed concatenated output from 1st layer to the 2nd layer
        output2, state2 = self.rnn2(input2, state2)

        seq_probs = []
        seq_preds = []

        # By this point we have already fed input features (of 30 frames) to 1st layer of LSTM and padded concatenated
        # inputs to 2nd layer of LSTM

        if mode == 'train':
            for i in range(self.max_length - 1):
                # generate word embeddings using the i-th column
                current_words = self.embedding(target_variable[:, i])

                # optimize for GPU (applicable only when CUDA/GPU capability is present in the system)
                self.rnn1.flatten_parameters()
                self.rnn2.flatten_parameters()

                # input frame paddings to 1st layer for the last 3 time steps
                output1, state1 = self.rnn1(padding_frames, state1)

                # concatenate word embeddings with output from the 1st layer, use the 3rd dimension to concat
                input2 = torch.cat((output1, current_words.unsqueeze(1)), dim=2)
                output2, state2 = self.rnn2(input2, state2)


                logits = self.out(output2.squeeze(1))
                logits = F.log_softmax(logits, dim=1)
                seq_probs.append(logits.unsqueeze(1))

            # concat values in 2nd dimension - which contains probabilities
            seq_probs = torch.cat(seq_probs, 1)

        elif mode == 'test':
            current_words = self.embedding(Variable(torch.LongTensor([self.sos_id] * batch_size)).cuda())
            for i in range(self.max_length - 1):
                # optimize for GPU (applicable only when CUDA/GPU capability is present in the system)
                self.rnn1.flatten_parameters()
                self.rnn2.flatten_parameters()

                output1, state1 = self.rnn1(padding_frames, state1)
                input2 = torch.cat((output1, current_words.unsqueeze(1)), dim=2)
                output2, state2 = self.rnn2(input2, state2)
                logits = self.out(output2.squeeze(1))
                logits = F.log_softmax(logits, dim=1)
                seq_probs.append(logits.unsqueeze(1))
                _, preds = torch.max(logits, 1)
                current_words = self.embedding(preds)
                seq_preds.append(preds.unsqueeze(1))

            seq_probs = torch.cat(seq_probs, 1)
            seq_preds = torch.cat(seq_preds, 1)

        else:
            raise RuntimeError(f"Unknown mode: {mode}")

        return seq_probs, seq_preds
