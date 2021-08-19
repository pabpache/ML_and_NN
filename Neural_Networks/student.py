#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""
######################################################################
### Students indentification
### Matias Morales z5216410
### Pablo Pacheco z5222810
### Group ID g023722
################################################
'''
##################################################################################

Our network structure was the following:  For rating: LSTM -> linear_layer -> relu_function  -> linear_layer -> sigmoid_function,
for Business category: LSTM -> linear_layer -> relu_function  -> linear_function. We used pack_padded to avoid making a prediction
with the padded part of the vectors (the vectors were padded because they are processed in batches, so they can have the same length).
LSTM was chosen over a simple RNN because it is needed that the network considers the first words of a review when is reading the last ones.
Also, the LSTM is a bidirectional network, this means that it is considered a prediction reading the review from left to right and another
from right to left. We used one output layer with one node for doing the binary classification (positive or negative) and another output layer
with 5 nodes for the multi-category classification (the type of business). In the one-node output layer, we used a sigmoid as an activation
function (which works well for binary classification). Meanwhile, in the five-nodes output layer, we don’t use the activation function because
we are going to use cross-entropy as a loss function and this function in PyTorch uses log_softmax, so we don’t need to apply log_softmax
(which would be the recommended activation function for multi-category classification). As loss function, it was used a mix. We used cross-entropy
for the multi-category classification which is the recommended loss function for this case, meanwhile, it was used binary cross-entropy to classify
the rating. About the convertNetOutput function, this was made to transform the output of the network for rating, which was a probability. In rating
output for values higher than 0.5 we assigned 1 and other values 0 because is a binary classification. While for category output we use argmax because
it returns the indices of the maximum value for each tensor, in this scenario the labels of business category is the same that the index values. 

Regarding training decisions, we choose the vector dimension of 300 instead of other values because we got better results. We tested changing different
hyperparameters such as hidden size (hidden layer of LSTM), dropout, number of layers, learning rate, optimizers, batch size, number of outputs of the first
linear layer and different combinations of linear layers after LSTM. Each tested combination was run 5 times because of the randomness of Neural Networks.
Adam was the optimizer chosen instead of SGD because Adam gets better performance than SGD, this can be explained because Adam uses Root Mean Square Propagation
and Stochastic Gradient Descent using a cumulative history of gradients which can take advantage of momentum compared with SGD. SGD iteratively updates model
parameters in the negative gradient direction and Adam employs adaptive per-parameter learning rates which can explain the reason for SGD converges much slower
than Adam. We used the following values where we got the best results: the batch size of 64, dropout rate 0.5 to avoid overfitting, the epoch was maintained by default,
100 as the output of the first linear layer, the number of hidden layers of 250 and the number of layers equal to 3.  In this scenario, we added 0.001 by 0.001
at each attempt and we detected that the model was worst increasing the learning rate of Adam were with a learning rate of 0.010 got around 20% accuracy and the
best results were around 0.001. This is the reason why we chose the learning rate of Adam of 0.001.

#################################################################################
'''


import torch

import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
# import numpy as np
# import sklearn

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """

    processed = sample.split()

    return processed

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """

    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch


# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
# stopWords=ENGLISH_STOP_WORDS
stopWords = {}
wordVectors = GloVe(name='6B', dim=300)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """

    ratingOutput = torch.tensor([1 if x > 0.5 else 0 for x in ratingOutput ]).to(device)
    categoryOutput=categoryOutput.argmax(dim=1)

    return ratingOutput, categoryOutput

################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()

         #  LSTM
        self.input_size = 300
        self.hidden_size = 250
        self.num_layers = 3
        
        self.lstm = tnn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = self.num_layers, batch_first = True, bidirectional = True)
        self.Linear_1 = tnn.Linear(self.hidden_size*2, 100)  
        self.Relu = tnn.ReLU()
        self.sig=tnn.Sigmoid()
        self.Linear_2 = tnn.Linear(100, 1) # 0，1
        self.Linear_3=tnn.Linear(100,5) # 0 1 2 3 4
        self.dropout=tnn.Dropout(0.5)



    def forward(self, input, length):

          #pack sequence
        if torch.cuda.is_available():
          packed_embedded = tnn.utils.rnn.pack_padded_sequence(input.cpu(), length.cpu(), batch_first=True).to(device)
        else: 
          packed_embedded = tnn.utils.rnn.pack_padded_sequence(input, length, batch_first=True)
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        x= self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        x = self.Linear_1(x)
        x = self.Relu(x)
        
        rating = self.Linear_2(x)
        rating = self.sig(rating)
        
        category = self.Linear_3(x)
   
        return rating, category

        


class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()

        self.cross_loss = tnn.CrossEntropyLoss()
        self.binary_cross_entropy = tnn.BCELoss()


    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        # because ratingoutput float answer
        ratingTarget = ratingTarget.type(torch.FloatTensor).to(device)

        category_loss = self.cross_loss(categoryOutput, categoryTarget)

        rating_loss = self.binary_cross_entropy(torch.squeeze(ratingOutput), ratingTarget)

        return rating_loss + category_loss

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
batchSize = 64
epochs = 10

optimiser = toptim.Adam(net.parameters(), lr=0.001)


