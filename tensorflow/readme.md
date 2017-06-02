# Tensorflow
```python
import tensorflow as tf
import numpy as np
import pandas as pd
```
Data are stored as tensor in Tensorflow, e.g. 
```python
3 # a rank 0 tensor; this is a scalar with shape []
[1. ,2., 3.] # a rank 1 tensor; this is a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
```

In Tensorflow, the basic elements are **node**. A node is an operation that takes some input tensors and yields an output tensor. Connecting various nodes gives a computational **graph**. One can start a **session** to run through the computational graph to provide the result.

A node can output a **constant** (no input), a **variable** (need initial value as input), or the **action** on other nodes (take other nodes as input), e.g.

```python
node1 = tf.constant(3.0, tf.float32) ＃ a constant
node2 = tf.constant(4.0) # also tf.float32 implicitly
node3 = tf.add(node1, node2) # can also use node3 = node1 + node2
```

Create a **session** to run through the graph of the three nodes (by passing the last node to the `sess.run()` function )
```python
sess = tf.Session()
print("sess.run(node3): ",sess.run(node3))
>>> sess.run(node3):  7.0
```

A **placeholder** node is promised to have a future input value of given type (sort of like a C++ function), e.g.
```python
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b 
print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))
>>> [ 3.  7.]
```

A **variable** node has an initial tensor as input, and behaves like a symbol until initializing (passing initial tensor to it) or a new input is assigned. In the following example we use variable nodes to build a linear model

```python
W = tf.Variable([.3], tf.float32) # .3 is the initial input
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

init = tf.global_variables_initializer()
sess.run(init)  # this initialize all global variables

print(sess.run(linear_model, {x:[1,2,3,4]}))
>>> [ 0.          0.30000001  0.60000002  0.90000004]

'''
Check the square root error using the initial values for parameters W and b
'''

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
>>> 23.66

'''
Improve the error by manually assigning new values to the parameters W and b
'''

fixW = tf.assign(W, [-1.]) # this define a new node, passing new value to the variable W
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb]) # run the nodes so they are given the values
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
>>> 0.0
```

Let's train the model instead of doing everything manually. Start from zero:

```python

# Model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # L2 loss

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
>>> W: [-0.9999969] b: [ 0.99999082] loss: 5.69997e-11
```

## Using `tf.contrib.learn`

Now we use the Tensorflow ML library `tf.contrib.learn` for the linear model build. 

```python
# Declare list of features. 
features = [tf.contrib.layers.real_valued_column("x", dimension=1)] # In this case only one real-valued feature. 

# Define the estimator; in this case use a linear regressor
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# Use `numpy_input_fn` to set up training dataset. 
# We have to tell the function how big the data size is (batch_size) 
# and how many times to go through the input data at most before stopping (num_epochs)
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4,
                                              num_epochs=1000)

# by pass the training data `input_fn` and steps to the estimator.fit() method, the estimator iterates through the training data multiple times.
# if steps > num_epochs, an "out of data" exception will be raised. 
estimator.fit(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did. 
print(estimator.evaluate(input_fn=input_fn))
>>> {'global_step': 1000, 'loss': 1.9650059e-11}
```

We can pass a customerized model to `tf.contrib.learn`, by passing the defined model to the `tf.contrib.learn.Estimator` method.

```python

def model(features, labels, mode):
  # Build a linear model and predict values
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  pred = W*features['x'] + b
  # Graph for the Loss term
  loss = tf.reduce_sum(tf.square(pred - labels))
  # Graph for Training 
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  # Output everything defined using ModelFnOps() method
  return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=pred,
      loss=loss,
      train_op=train)

#Pass the customerized model to estimator
estimator = tf.contrib.learn.Estimator(model_fn=model)
# define our data set
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, 4, num_epochs=1000)

# train
estimator.fit(input_fn=input_fn, steps=1000)
# evaluate our model
print(estimator.evaluate(input_fn=input_fn, steps=10))
```


