import numpy as np

#Extrapolates from a given seed sequence
def generate_from_seed(model, seed, sequence_length, data_variance, data_mean):
	seedSeq = seed.copy()
	output = []

	#The generation algorithm is simple:
	#Step 1 - Given A = [X_0, X_1, ... X_n], generate X_n + 1
	#Step 2 - Concatenate X_n + 1 onto A
	#Step 3 - Repeat MAX_SEQ_LEN times
	for it in xrange(sequence_length):
		seedSeqNew = model.predict(seedSeq) #Step 1. Generate X_n + 1
		#Step 2. Append it to the sequence
		if it == 0:
			for i in xrange(seedSeqNew.shape[1]):
				output.append(seedSeqNew[0][i].copy())
		else:
			output.append(seedSeqNew[0][seedSeqNew.shape[1]-1].copy()) 
		newSeq = seedSeqNew[0][seedSeqNew.shape[1]-1]
		newSeq = np.reshape(newSeq, (1, 1, newSeq.shape[0]))
		seedSeq = np.concatenate((seedSeq, newSeq), axis=1)

	#Finally, post-process the generated sequence so that we have valid frequencies
	#We're essentially just undo-ing the data centering process
	for i in xrange(len(output)):
		output[i] *= data_variance
		output[i] += data_mean
	return output
