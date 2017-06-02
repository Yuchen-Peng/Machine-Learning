# Tensorflow
```python
import tensorflow as tf
import numpy as np
import pandas as pd
```
Data are stored as tensor in Tensorflow. 

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

'''
Let's train the model instead of doing everything manually
'''
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([W, b]))
>>> [array([-0.9999969], dtype=float32), array([ 0.99999082], dtype=float32)] # the parameter value

print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]})) # the error with the new parameter values 
>>> 5.69997e-11
```
