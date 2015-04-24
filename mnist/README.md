Everything regarding supervised learning on mnist

The data folder contains the H2O datasets, provided for the H20 training. I just wanted to compare the results obtained using H20 and Torch, while being sure that the data is exactly the same.

- h2o_ffnn.lua: A simple feedforward neural net designed primarily to make a first working net with Torch...
The version (as of 04/23/2015) is simply taking the 784 inputs of the h2o mnist, and passes it through 5 layers ({1024, 2048, 2048, 1024, 512} units), to 10 output units, then a LogSoftMax(), for using NLL criterion.
I used Dropout (p=0.5) for each layer, but no dropout on the inputs (that can be done for noisy inputs). And that's it.
Using only 5000 training samples, and 10 epochs, it achieves a 92.3% accuracy. This is bad regarding the current record, but pretty good when taking into account that I was just messing around to make a first working model. And, note that dropout is quite good for that regularization thing. 
From there, I will try to make this working on GPU, to speed up these tests. The goal is in the end to get something close to the 99% accuracy (at least)
- h2o_ffnn_gpu.lua: Pretty much the same than the previous one, only that it uses cutorch and cunn to run on GPU. It allows to use more data, and larger configurations. With the same config as described above, we reach a 98.25% accuracy on the test set, within 15 epochs. More doesn't seem to help much (or I would just have to run it for extremely longer, which is not possible at the time).
With a larger configuration:
