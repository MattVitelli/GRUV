from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import os
import nn_utils.network_utils as network_utils
import gen_utils.seed_generator as seed_generator
import gen_utils.sequence_generator as sequence_generator
from data_utils.parse_files import *
import config.nn_config as nn_config

config = nn_config.get_neural_net_configuration()
sample_frequency = config['sampling_frequency']
inputFile = config['model_file']
model_basename = config['model_basename']
cur_iter = 25
model_filename = model_basename + str(cur_iter)
output_filename = './generated_song.wav'

#Load up the training data
print ('Loading training data')
#X_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
#y_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
#X_mean is a matrix of size (num_frequency_dims,) containing the mean for each frequency dimension
#X_var is a matrix of size (num_frequency_dims,) containing the variance for each frequency dimension
X_train = np.load(inputFile + '_x.npy')
y_train = np.load(inputFile + '_y.npy')
X_mean = np.load(inputFile + '_mean.npy')
X_var = np.load(inputFile + '_var.npy')
print ('Finished loading training data')

#Figure out how many frequencies we have in the data
freq_space_dims = X_train.shape[2]
hidden_dims = config['hidden_dimension_size']

#Creates a lstm network
model = network_utils.create_lstm_network(num_frequency_dimensions=freq_space_dims, num_hidden_dimensions=hidden_dims)
#You could also substitute this with a RNN or GRU
#model = network_utils.create_gru_network()

#Load existing weights if available
if os.path.isfile(model_filename):
	model.load_weights(model_filename)
else:
	print('Model filename ' + model_filename + ' could not be found!')

print ('Starting generation!')
#Here's the interesting part
#We need to create some seed sequence for the algorithm to start with
#Currently, we just grab an existing seed sequence from our training data and use that
#However, this will generally produce verbatum copies of the original songs
#In a sense, choosing good seed sequences = how you get interesting compositions
#There are many, many ways we can pick these seed sequences such as taking linear combinations of certain songs
#We could even provide a uniformly random sequence, but that is highly unlikely to produce good results
seed_len = 1
seed_seq = seed_generator.generate_copy_seed_sequence(seed_length=seed_len, training_data=X_train)

max_seq_len = 10; #Defines how long the final song is. Total song length in samples = max_seq_len * example_len
output = sequence_generator.generate_from_seed(model=model, seed=seed_seq, 
	sequence_length=max_seq_len, data_variance=X_var, data_mean=X_mean)
print ('Finished generation!')

#Save the generated sequence to a WAV file
save_generated_example(output_filename, output, sample_frequency=sample_frequency)