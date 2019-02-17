import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  # for each training example, we want to find the loss and the gradient.
  for i in range(num_train):
    # Score of each class output = Input Picture * some W matrix.
    # f(x) = X dot W
    scores = X[i].dot(W)
    # y[i] is the correct class, so we want to extract our model's score of that class.
    correct_class_score = scores[y[i]]

    # for each class, we want to compute the SVM hinge thing, where if the
    # score of the class is greater than some margin (1), we would add that
    # to the loss. i.e. if the score of a wrong class is greater than 1
    # add that to the total loss/'badness' of the data.

    # https://github.com/lightaime/cs231n/blob/master/assignment1/cs231n/classifiers/linear_svm.py
    # http://cs231n.github.io/optimization-1/#gradcompute for explanation
    # tldr; if we take the gradient of W (slope towards the right direction)
    # we take the partial derivatives/Jacobian
    # this is w_1, w_2, ... w_10 for each class. (10 dimensions)
    # but there are basically two possibilities: w_i (correct) or w_j (incorrect)
    # so we get (del) df/dw_j = x_i and df/dw_i = -x_i
    # dW, the slope, is the gradient * function itself, so we get
    # that dW should be the slope of each training image towards the right direction.
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # add -x_i to the correct column and x_i to the incorrect column if margin > 0
        dW[:,j] += X[i].T
        dW[:,y[i]] += -X[i].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  num_train = X.shape[0]
  num_classes = W.shape[1]

  # compte the score for picture in being each class.
  scores = X.dot(W) # 500x3072 * 3072 * 10 = 500x10
  # get the correct class score for each training example
  correct_class_scores = []
  for i in range(num_train):
      correct_class_scores.append(np.array([scores[i][y[i]]]))
  correct_class_scores = np.array(correct_class_scores)

  margins = np.maximum(0, scores - correct_class_scores + 1)
  for i in range(num_train):
      margins[i][y[i]] = 0
  # loss over entire data
  loss = np.sum(margins) / num_train
  # loss with regularization (this is tiny )
  loss += reg * np.sum(W * W)

  # https://github.com/lightaime/cs231n/blob/master/assignment1/cs231n/classifiers/linear_svm.py
  # printing out the output of dW[0] on each iteration of the naiive helps
  # this make sense.
  coeff_mat = np.zeros((num_train, num_classes))
  coeff_mat[margins > 0] = 1
  coeff_mat[range(num_train), list(y)] = 0
  coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)

  dW = X.T.dot(coeff_mat) / num_train + reg * W

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
