# GRUV
GRUV is a Python project for algorithmic music generation using recurrent neural networks.

Note: This code works with Keras v. 0.1.0, later versions of Keras may not work.

For a demonstration of our project on raw audio waveforms (as opposed to the standard MIDI), see here: https://www.youtube.com/watch?v=0VTI1BBLydE

Copyright (C) 2015 Matt Vitelli matthew.vitelli@gmail.com and Aran Nayebi aran.nayebi@gmail.com

# Dependencies
In order to use GRUV, you will first need to install the following dependencies:

Theano: http://deeplearning.net/software/theano/#download

Keras: https://github.com/fchollet/keras.git

NumPy: http://www.numpy.org/

SciPy: http://www.scipy.org/

LAME (for MP3 source files): http://lame.sourceforge.net/ 

SoX (for FLAC source files): http://sox.sourceforge.net/

Once that's taken care of, you can try training a model of your own as follows:
# Step 1. Prepare the data
Copy your music into ./datasets/YourMusicLibrary/ and type the following command into Terminal:
>    python convert_directory.py

This will convert all mp3s in ./datasets/YourMusicLibrary/ into WAVs and convert the WAVs into a useful representation for the deep learning algorithms.

# Step 2. Train your model
At this point, you should have four files named YourMusicLibraryNP_x.npy, YourMusicLibraryNP_y.npy, YourMusicLibraryNP_var.npy, and YourMusicLibraryNP_mean.npy.

YourMusicLibraryNP_x contains the input sequences for training
YourMusicLibraryNP_y contains the output sequences for training
YourMusicLibraryNP_mean contains the mean for each feature computed from the training set
YourMusicLibraryNP_var contains the variance for each feature computed from the training set

You can train your very first model by typing the following command into Terminal:
>    python train.py

Training will take a while depending on the length and number of songs used
If you get an error of the following form:
Error allocating X bytes of device memory (out of memory). Driver report Y bytes free and Z bytes total
you must adjust the parameters in train.py - specifically, decrease the batch_size to something smaller. If you still have out of memory errors, you can also decrease the hidden_dims parameter in train.py and generate.py, although this will have a significant impact on the quality of the generated music.

# Step 3. Generation
After you've finished training your model, it's time to generate some music!
Type the following command into Terminal:
>    python generate.py

After some amount of time, you should have a file called generated_song.wav

Future work:
Improve generation algorithms. Our current generation scheme uses the training / testing data as a seed sequence, which tends to produce verbatum copies of the original songs. One might imagine that we could improve these results by taking linear combinations of the hidden states for different songs and projecting the combinations back into the frequency space and using those as seed sequences. You can find the core components of the generation algorithms in gen_utils/seed_generator.py and gen_utils/sequence_generator.py
