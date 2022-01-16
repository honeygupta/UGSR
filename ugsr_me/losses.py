"""
Project: Unaligned Guided Thermal Image Super-Resolution
Author: Honey Gupta (hn.gpt1@gmail.com)

"""
import tensorflow as tf
from tensorflow.python.ops import array_ops


def rec_loss(output, gt):
	return tf.losses.absolute_difference(gt, output)


def l0_norm(img):
	return tf.cast(tf.count_nonzero(image_gradients(img)), tf.float32)


def image_gradients(image):
	image_shape = array_ops.shape(image)
	batch_size, height, width, depth = array_ops.unstack(image_shape)
	dy = image[:, 1:, :] - image[:, :-1, :, :]
	dx = image[:, :, 1:, :] - image[:, :, :-1, :]
	shape = array_ops.stack([batch_size, 1, width, depth])
	dy = array_ops.concat([dy, array_ops.zeros(shape, image.dtype)], 1)
	dy = array_ops.reshape(dy, image_shape)
	shape = array_ops.stack([batch_size, height, 1, depth])
	dx = array_ops.concat([dx, array_ops.zeros(shape, image.dtype)], 2)
	dx = array_ops.reshape(dx, image_shape)
	return dy, dx


def gradient_loss(output, gt):
	g1 = image_gradients(output)
	g2 = image_gradients(gt)
	return tf.losses.absolute_difference(g1, g2)
