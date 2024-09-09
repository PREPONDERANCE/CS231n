# Optimization

## Computing the Gradients

There're two ways to compute the gradients: numerical and analytical.

### Numerical

Recall when we first get our hands on calculating the derivative of a
function, we define the following method:

$$
f'(x) = \lim_{h \to 0} \dfrac{f(x+h) - f(x)}{h}
$$

$h$ is typically a small enough value since strictly speaking, the
mathematical definition requires $h$ to be close enough to $0$.

Moving it to broader dimensions, we can calculate the partial derivative
by iterating through each dimension using the above method.

**Example code:**

```Python
def eval_numerical_gradient(f, x):
  """
  a naive implementation of numerical gradient of f at x
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """

  fx = f(x) # evaluate function value at original point
  grad = np.zeros(x.shape)
  h = 0.00001

  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    old_value = x[ix]
    x[ix] = old_value + h # increment by h
    fxh = f(x) # evalute f(x + h)
    x[ix] = old_value # restore to previous value (very important!)

    # compute the partial derivative
    grad[ix] = (fxh - fx) / h # the slope
    it.iternext() # step to next dimension

  return grad
```

### Analytical

The analytical method basically precalculates a formula for the
actual gradients and implement that function with vectorized code.

It's important, however, to apply a **gradient check** on the
analytical results to make sure our calculation is correct. This
so-called check simply compare the analytical results with the
numerical results for equality.
