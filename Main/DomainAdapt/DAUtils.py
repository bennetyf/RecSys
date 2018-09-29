from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops


# Flip Gradient Layer
class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y

# Generate the corresponding embedding given the pre-trained input user rating vector
# class EmbeddingGenerator(object):
#     def __init__(self, sess, restore_path):
#         self.res_path = restore_path
#         saver = tf.train.Saver()
#         saver.restore(sess, restore_path)
#
#     def __call__(self, raw_vector):



flip_gradient = FlipGradientBuilder()
