It is recommended to intstall the tensorflow package via virtual environment from conda. 

Create a conda environment named `tensorflow`

```
conda create -n tensorflow
```

Activate the environment

```
$ source activate tensorflow # Mac
C:> activate tensorflow # Windows
```

Install package
```
(tensorflow)$ pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.1.0-py2-none-any.whl # Mac
(tensorflow)C:> pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.1.0-cp35-cp35m-win_amd64.whl # windows
```

Validate: activate the environment, then

```
$ python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
```

To deactivate the environment
```
(tensorflow)$ deactivate 
```
