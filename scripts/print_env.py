import sys
import tensorflow as tf
import importlib, importlib.util

print('python executable:', sys.executable)
print('tensorflow version:', tf.__version__)
print('tensorflow module file:', getattr(tf, '__file__', None))

try:
    import keras
    print('keras package version:', getattr(keras, '__version__', None))
    print('keras module file:', getattr(keras, '__file__', None))
except Exception as e:
    print('keras import failed or not installed:', e)

spec = importlib.util.find_spec('keras')
print('importlib.find_spec("keras"):', spec)

spec_tf_keras = importlib.util.find_spec('tensorflow.keras')
print('importlib.find_spec("tensorflow.keras"):', spec_tf_keras)
