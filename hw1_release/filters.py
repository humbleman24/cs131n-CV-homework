"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pad_height = Hk // 2
    pad_width = Wk // 2
    # I pad the image first to keep the dimension of the output convoluded image
    image = zero_pad(image,pad_height,pad_width)
    for i in range(pad_height, Hi+pad_height):
        for j in range(pad_width, Wi+pad_width):
            sum = 0
            for k in range(Hk):
                for l in range(Wk):
                    # two things need to be consider
                    # kernel should be flipped
                    # how to match the pixel in the image
                    #                          + means flip             () match pixel
                    sum += kernel[k][l] * image[i + (pad_height-k)][j + (pad_width-l)]
            out[i-pad_height][j-pad_width] = sum

    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    zero_row = np.zeros([pad_height,W])
    zero_column = np.zeros([H+2*pad_height,pad_width])
    out = np.vstack([zero_row,image])
    out = np.vstack([out,zero_row])
    out = np.hstack([zero_column,out])
    out = np.hstack([out, zero_column])
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    fliped_kernel = np.zeros_like(kernel)

    ### YOUR CODE HERE
    pad_height = Hk // 2
    pad_width = Wk // 2
    image = zero_pad(image,pad_height,pad_width)
    # flip kernel
    fliped_kernel = np.flip(kernel)
    # convolution
    for i in range(0, Hi):
        for j in range(0, Wi):
            sum = np.sum(image[i:i+Hk,j:j+Wk] * fliped_kernel)
            out[i][j] = sum
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pad_height = Hk // 2
    pad_width = Wk // 2
    image = zero_pad(image,pad_height,pad_width)
    # flip kernel
    fliped_kernel = np.flip(kernel)
    mat = np.zeros((Hi*Wi,Hk*Wk))
    for i in range(Hi*Wi):
        row = i // Wi
        column = i % Wi
        mat[i] = image[row:row+Hk,column:column+Wk].reshape(1,Hk*Wk)
    out = mat.dot(fliped_kernel.reshape(Hk*Wk,1)).reshape(Hi,Wi)
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    out = np.zeros((Hf,Wf))
    ### YOUR CODE HERE
    pad_height = Hg // 2
    pad_width = Wg // 2
    image = zero_pad(f,pad_height,pad_width)
    for i in range(Hf):
        for j in range(Wf):
            out[i][j] = np.sum(image[i:i+Hg,j:j+Wg] * g)
    ### END YOUR CODE
    '''if take advantage of conv_fast, just fliped the g filter first
        and than pass to the conv_fast function
    g = np.flip(g)
    out = conv_fast(f,g)    
    '''


    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g = g - np.mean(g)
    g = np.flip(g)
    out = conv_fast(f,g)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    Hf, Wf = f.shape
    Hg, Wg = g.shape
    out = np.zeros((Hf,Wf))
    ### YOUR CODE HERE
    pad_height = Hg // 2
    pad_width = Wg // 2
    image = zero_pad(f,pad_height,pad_width)
    g = normalize(g)    
    for i in range(Hf):
        for j in range(Wf):
            out[i][j] = np.sum(normalize(image[i:i+Hg,j:j+Wg]) * g)
    ### END YOUR CODE
    return out

def normalize(array):
    # this function is used to normalize the patch of image and kernel
    # improve the robustness to the light change in cross-correlation
    mean = np.mean(array)
    std = np.std(array)
    # it should use the standard deviation
    #var = np.var(array)

    return (array-mean)/std
