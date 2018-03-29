# CreatorofNeurals
Creates instances of simple neural networks based on the parameters provided.
Parameters are: 

>Layer data in the form of a list 

>Input layer in the form of a list


ip():- takes in data for the input layer in the form of a list containg numbers

compute():- computes the final(output) layer of the neural network.

ncomp():- Computes the final layer for given parameters

enter():- takes in a list containing numpy arrays, where list[0] is the list containing lists of weights for each layer where each list of weights (with index n) corresponds to the n + 1th layer in the list containing the layers and list[1] is the list containg the biases for each link.

exunt():- outputs a list containg the weights for each link.

backprop():- takes in set_list and corrects the weights and biases.

set_list format should be [[list of inputs], [Correct value for each input]] 

Dependencies:

NumPy(http://www.numpy.org/)

This was my second try at hacking together a neural network, so expect inefficencies.
Edit: Works but requires some changes to the gradient descent algorithm
