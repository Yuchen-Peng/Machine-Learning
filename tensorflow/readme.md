# Tensorflow
```
import tensorflow as tf
```
Data are stored as tensor in Tensorflow. 

In Tensorflow, the basic elements are **node**. A node is an operation that takes some input tensors and yields an output tensor. Connecting various nodes gives a computational **graph**. One can start a **session** to run through the computational graph to provide the result.

A node can output a **constant** (no input), a **variable** (need initial value as input), or the **action** on other nodes (take other nodes as input), e.g.

```
node1 = tf.constant(3.0, tf.float32) ＃ a constant
node2 = tf.constant(4.0) # also tf.float32 implicitly
node3 = tf.add(node1, node2) # can also use node3 = node1 + node2
```

Create a **session** to run through the graph of the three nodes:
```
sess = tf.Session()
print("sess.run(node3): ",sess.run(node3))
>>> sess.run(node3):  7.0
```

A **placeholder** node is promised to have a future input value of given type (sort of like a C++ function), e.g.
```
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b 
print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))
>>> [ 3.  7.]
```
