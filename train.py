from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import os
import nn_utils.network_utils as network_utils

#TODO - Make these command line arguments
inputFile = './datasets/YourMusicLibraryNP'
cur_iter = 0
model_basename = './YourMusicLibraryNPWeights'
model_filename = model_basename + str(cur_iter)

#Load up the training data
print ('Loading training data')
#X_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
#y_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
X_train = np.load(inputFile + '_x.npy')
y_train = np.load(inputFile + '_y.npy')
print ('Finished loading training data')

#TODO - Since these parameters are shared by train.py and generate.py, 
#they should likely be stored somewhere so there's no duplication of code
#Figure out how many frequencies we have in the data
freq_space_dims = X_train.shape[2]

#Number of hidden dimensions.
#For best results, this should be >= freq_space_dims, but most consumer GPUs can't handle large sizes
hidden_dims = 1024

#Creates a lstm network
model = network_utils.create_lstm_network(num_frequency_dimensions=freq_space_dims, num_hidden_dimensions=hidden_dims)
#You could also substitute this with a RNN or GRU
#model = network_utils.create_gru_network()

#Load existing weights if available
if os.path.isfile(model_filename):
	model.load_weights(model_filename)

num_iters = 25 			#Number of iterations for training
epochs_per_iter = 25	#Number of iterations before we save our model
batch_size = 1			#Number of training examples pushed to the GPU per batch.
						#Larger batch sizes require more memory, but training will be faster
print ('Starting training!')
while cur_iter < num_iters:
	print('Iteration: ' + str(cur_iter))
	#We set cross-validation to 0,
	#as cross-validation will be on different datasets 
	#if we reload our model between runs
	#The moral way to handle this is to manually split 
	#your data into two sets and run cross-validation after 
	#you've trained the model for some number of epochs
	history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs_per_iter, verbose=1, validation_split=0.0)
	cur_iter += epochs_per_iter
print ('Training complete!')
model.save_weights(model_basename + str(cur_iter))