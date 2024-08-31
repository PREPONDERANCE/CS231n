# SVM

## Forward

The loss of each class is calculated as below:

$$
\begin{equation}
L = \sum_{j}^{j \neq y_i} \max(0, S_j - S_{y_i} + \delta)
\end{equation}
$$

An example will better illustrate the process.

Suppose there are 3 classes and each score is $s_1$, $s_2$, $s_3$ and
this image belongs to label $1$, i.e. $y_i = 1$. Also, $\delta$ is set
to 1 as default.

$$
L = max(s_2 - s_1 + 1, 0) + max(s_3 - s_1 + 1, 0)
$$

Notice that we didn't take $max(s_1 - s_1 + 1, 0)$ since $s_1$ is our
target label.

## SVM Nature

SVM essentially wants the loss of correct class to be larger than the
incorrect classes by at least $\delta$. If this is not the case, we
will accumulate loss.

## Backward

When it comes to calculating the gradients, we need to write down every
single equation and with the help of chain rule, we connect each separate
part together.

$$
\begin{align}
L &= \dfrac{1}{N} \sum_{i=1}^{N} L_i \\
  &= \dfrac{1}{N} \sum_{i=1}^{N} (\sum_{j}^{j \neq y_i} \max(0, s_j - s_{y_i} + 1))
\end{align}
$$

To calculate the gradients:

$$
\begin{align}
\dfrac{\partial L}{\partial W} &= \dfrac{1}{N} \sum_{i=1}^{N} \dfrac{\partial L_i}{\partial W} \\
\end{align}
$$

Abstract it:

$$
\begin{align}
\dfrac{\partial L_i}{\partial W} &= \sum_{k=1}^{C} \dfrac{\partial L_i}{\partial s_k} \times \dfrac{\partial s_k}{\partial W}
\end{align}
$$

Furthermore:

$$
\begin{align}
\dfrac{\partial L_i}{\partial s_k} &= \dfrac{\partial \sum_{j}^{j \neq y_i} \max(0, s_j - s_{y_i} + 1)}{\partial s_k} \\
                                   &= \sum_{j}^{j \neq y_i} \dfrac{\partial \max(0, s_j - s_{y_i} + 1)}{\partial s_k} \\
                                   &= \begin{cases}
                                      0 & \text{if } j = y_i \text{ or } k \neq j \text{ or } s_j - s_{y_i} + 1 \leq 0 \\
                                      1 & \text{otherwise }
                                      \end{cases}
\end{align}
$$

in which $s_k$ is short for $s_k - s_{y_i} + 1$. This may seem a little
unintuitive. Just treat $s_j - s_{y_i} + 1$ as a whole and the entire
expression will make a lot more sense.

Furthermore:

$$
\begin{align}
\dfrac{\partial s_k}{\partial W} &= \begin{bmatrix}
                                    \dfrac{\partial s_k}{\partial W_{11}} & \dfrac{\partial s_k}{\partial W_{12}} & \dots  & \dfrac{\partial s_k}{\partial W_{1C}} \\
                                    \vdots & \vdots & \ddots & \vdots \\
                                    \dfrac{\partial s_k}{\partial W_{D1}} & \dfrac{\partial s_k}{\partial W_{D2}} & \dots  & \dfrac{\partial s_k}{\partial W_{DC}}
                                    \end{bmatrix}
\end{align}
$$

At each matrix slot, the value of $\dfrac{\partial s_k}{\partial W_{mn}}$
depends on whether $s_k$ is 0.

If it is, the derivative is 0 for sure.

If not,

$$
\begin{align}
s_k &= s_k - s_{y_i} + 1 \\
    &= \mathbf{X} \cdot \mathbf{W_k} - \mathbf{X} \cdot \mathbf{W_{y_i}} + 1
\end{align}
$$

**Note**, the $W_k$ here means the kth column of the weight matrix.

Thus, (suppose $s_k$ is not 0), $\dfrac{\partial s_k}{\partial W_{mn}} = X_n$

## Code

The theory is a mind-boggling. Let's walk through the code along
with the theories we've just cracked.

### Naive Approach

```Python
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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]
                loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result, you may need to modify some of the   #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Average the gradient
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
```

### Vectorized Approach

```Python
def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = np.dot(X, W)
    scores = np.maximum(0, scores - scores[range(len(scores)), y].reshape(-1, 1) + 1)
    scores[range(len(scores)), y] = 0

    loss = np.sum(scores) / X.shape[0]
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dvalues = (scores > 0).astype(float)
    dvalues[range(len(dvalues)), y] -= np.sum(dvalues, axis=1)
    dW = np.dot(X.T, dvalues) / X.shape[0]

    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

```
