import numpy as np
from random import shuffle
import math

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  num_classes = W.shape[1]

  # Softmax Function:
  # L_i = -log(\frac{e^{s_y_i}}{\sum_j{e^{s_j}})

  # # for each training img, compute the loss given by softmax function
  # for i in range(num_train):
  #   # compute the scores
  #   scores_i = X[i].dot(W)
  #   # exponentiate them
  #   scores_exp = np.exp(scores_i)
  #   bot = np.sum(scores_exp)
  #
  #   loss_i = -1 * math.log(score_y / bot )
  #   loss += loss_i
  #
  #   for j in range(num_classes):
  #     if y[i] == j:
  #       dW[:,j] += loss_i * (1 - loss_i)
  #     else:
  #       print('hi')
  #
  #
  # loss /= num_train
  # loss += 0.5 * reg * np.sum(W*W)


  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  
  score += - np.max(score , axis=0)
  exp_score = np.exp(score) # matric exponientiel score
  sum_exp_score_col = np.sum(exp_score , axis = 0) # sum des expo score pr chaque column

  loss = np.log(sum_exp_score_col)
  loss = loss - score[y,np.arange(num_train)]
  loss = np.sum(loss) / float(num_train) + 0.5 * reg * np.sum(W*W)

  Grad = exp_score / sum_exp_score_col
  Grad[y,np.arange(num_train)] += -1.0
  dW = Grad.dot(X.T) / float(num_train) + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
