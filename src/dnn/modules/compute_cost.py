def compute_cost(AL, Y, layers_dims, parameters, lambd=0):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
    parameters -- python dictionary containing parameters of the model

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]
    L = len(layers_dims)           # number of layers in the network
    L2_regularization_cost = 0.

    if lambd != 0:
        for l in range(1, L):
            Wl = parameters['W' + str(l)]
            L2_regularization_cost += np.sum(np.square(Wl))
        L2_regularization_cost *= (1 / m) * (lambd / 2)
    
   

    # Compute loss from aL and y.
    cost = (-1 / m) * (np.dot(Y, np.log(AL.T)) + np.dot(1-Y, np.log((1-AL).T))) + L2_regularization_cost
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    
    return cost