# ================================================
# Creating a Recurrent Neural Network (RNN)
# ================================================

import torch.nn as nn

import src.common.tools as tools
import src.preprocess.dataset as dataset


class ClassifyingCharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ClassifyingCharRNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.labels_uniq = dataset.alldata.labels_uniq

    def forward(self, line_tensor):
        rnn_out, hidden = self.rnn(line_tensor)
        output = self.h2o(hidden[0])
        output = self.softmax(output)

        return output

    def label_from_output(self, output, output_labels):
        top_n, top_i = output.topk(1)
        label_i = top_i[0].item()
        return output_labels[label_i], label_i

    def predict(self, text):
        input = tools.line_to_tensor(text)
        output = self(input)
        return self.label_from_output(output, self.labels_uniq)


n_hidden = 128
classifying_rnn = ClassifyingCharRNN(
    tools.n_letters, n_hidden, len(dataset.alldata.labels_uniq))
# print(classifying_rnn)


def label_from_output(output, output_labels):
    top_n, top_i = output.topk(1)
    label_i = top_i[0].item()
    return output_labels[label_i], label_i

# input = tools.line_to_tensor("Takahiro")
# output = classifying_rnn(input)
# print(output)
# print(label_from_output(output, dataset.alldata.labels_uniq))
