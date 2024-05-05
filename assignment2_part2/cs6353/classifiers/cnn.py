from builtins import object
import numpy as np

from cs6353.layers import *
from cs6353.layer_utils import *


class ConvNet(object):
    """
   A simple convolutional network with the following architecture:

    [conv - bn - relu] x M - max_pool - affine - softmax
    
    "[conv - bn - relu] x M" means the "conv-bn-relu" architecture is repeated for
    M times, where M is implicitly defined by the convolution layers' parameters.
    
    For each convolution layer, we do downsampling of factor 2 by setting the stride
    to be 2. So we can have a large receptive field size.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=[32], filter_sizes=[7],
                 num_classes=10, weight_scale=1e-3, reg=0.0,use_batch_norm=True, hidden_dim=100,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer. It is a
          list whose length defines the number of convolution layers
        - filter_sizes: Width/height of filters to use in the convolutional layer. It
          is a list with the same length with num_filters
        - num_classes: Number of output classes
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - use_batch_norm: A boolean variable indicating whether to use batch normalization
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes

        ############################################################################
        # TODO: Initialize weights and biases for the simple convolutional         #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params.                                                 #
        #                                                                          #
        # IMPORTANT:                                                               #
        # 1. For this assignment, you can assume that the padding                  #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. You need to         #
        # carefully set the `pad` parameter for the convolution.                   #
        #                                                                          #
        # 2. For each convolution layer, we use stride of 2 to do downsampling.    #
        ############################################################################
        C, H, W = input_dim
        self.convNetCount = len(num_filters)
        self.fcnum = self.convNetCount+1
        self.num_filters = np.hstack((C, num_filters))

        for i in range(self.convNetCount):
            self.params['W{}'.format(i+1)] = np.random.randn(self.num_filters[i+1], self.num_filters[i], filter_sizes[i], filter_sizes[i]) * weight_scale        
            self.params['b{}'.format(i+1)] = np.zeros(self.num_filters[i+1])

        HP, WP = (H - 2)//2 + 1, (W - 2)//2 + 1 
        
        # print(self.num_filters[i+1], num_filters[self.fcnum-2], [H, W], [HP, WP], num_filters[self.fcnum-2] * HP * WP)

        self.params['W{}'.format(self.fcnum)] = np.random.randn(num_filters[self.fcnum-2] * H * W, hidden_dim) * weight_scale
        self.params['b{}'.format(self.fcnum)] = np.zeros(hidden_dim)

        self.params['W{}'.format(self.fcnum+1)] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b{}'.format(self.fcnum+1)] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        scores = None
        mode = 'test' if y is None else 'train'
        ############################################################################
        # TODO: Implement the forward pass for the simple convolutional net,       #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        prevInp = X
        conv_caches = [] 
        for i in range(self.convNetCount):
            conv_param = {"stride": 1, "pad": (self.filter_sizes[i] - 1) // 2}
            pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}
            W, b = self.params["W{}".format(i+1)], self.params["b{}".format(i+1)]
            relu_out, conv_cache = conv_relu_forward(prevInp, W, b, conv_param)#, pool_param)
            conv_caches.append(conv_cache)
            prevInp = relu_out
        
        # print(prevInp.shape)
        A2, fc1_cache = affine_relu_forward(prevInp, self.params["W{}".format(self.fcnum)], self.params["b{}".format(self.fcnum)])

        scores, fc2_cache = affine_forward(A2, self.params["W{}".format(self.fcnum+1)], self.params["b{}".format(self.fcnum+1)])

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the simple convolutional net,      #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, softmax_grad = softmax_loss(scores, y)

        for i in range(self.convNetCount):
            W = self.params["W{}".format(i+1)]
            loss += 0.5 * self.reg * np.sum(W * W)
        loss += 0.5 * self.reg * np.sum(self.params["W{}".format(self.fcnum)])
        loss += 0.5 * self.reg * np.sum(self.params["W{}".format(self.fcnum+1)])

        # print('W{}'.format(self.fcnum+1))
        dout, grads['W{}'.format(self.fcnum+1)], grads['b{}'.format(self.fcnum+1)] = affine_backward(softmax_grad, fc2_cache)
        grads['W{}'.format(self.fcnum+1)] += self.reg * self.params["W{}".format(self.fcnum+1)]

        # print('W{}'.format(self.fcnum))
        dout, grads['W{}'.format(self.fcnum)], grads['b{}'.format(self.fcnum)] = affine_relu_backward(dout, fc1_cache)
        grads['W{}'.format(self.fcnum)] += self.reg * self.params["W{}".format(self.fcnum)]

        # for c in conv_cache:
        #     for item in c:
        #         print(item.shape)
        for i in range(self.convNetCount-1, -1, -1):
            # print('W{}'.format(i+1))
            dout, grads["W{}".format(i+1)], grads["b{}".format(i+1)] = conv_relu_backward(dout, conv_caches[i])
            grads['W{}'.format(i+1)] += self.reg * self.params["W{}".format(i+1)]

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
