import torch
import tensorflow as tf


print(torch.cuda.is_available())
print(tf.config.list_physical_devices('GPU'))