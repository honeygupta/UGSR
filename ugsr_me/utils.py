"""
Project: Unaligned Guided Thermal Image Super-Resolution
Author: Honey Gupta (hn.gpt1@gmail.com)

"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops

from tensorflow.python.ops import math_ops


def diff_x(input, r):
	assert input.shape.ndims == 4

	left = input[:, :, r:2 * r + 1]
	middle = input[:, :, 2 * r + 1:] - input[:, :, :-2 * r - 1]
	right = input[:, :, -1:] - input[:, :, -2 * r - 1:    -r - 1]

	output = tf.concat([left, middle, right], axis=2)

	return output


def diff_y(input, r):
	assert input.shape.ndims == 4

	left = input[:, :, :, r:2 * r + 1]
	middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1]
	right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1:    -r - 1]

	output = tf.concat([left, middle, right], axis=3)

	return output


def box_filter(x, r):
	assert x.shape.ndims == 4

	return diff_y(tf.cumsum(diff_x(tf.cumsum(x, axis=2), r), axis=3), r)


def guided_filter(x, y, r, eps=1e-8, nhwc=True):
	assert x.shape.ndims == 4 and y.shape.ndims == 4

	# data format
	if nhwc:
		x = tf.transpose(x, [0, 3, 1, 2])
		y = tf.transpose(y, [0, 3, 1, 2])

	# shape check
	x_shape = tf.shape(x)
	y_shape = tf.shape(y)

	assets = [tf.assert_equal(x_shape[0], y_shape[0]),
	          tf.assert_equal(x_shape[2:], y_shape[2:]),
	          tf.assert_greater(x_shape[2:], 2 * r + 1),
	          tf.Assert(tf.logical_or(tf.equal(x_shape[1], 1),
	                                  tf.equal(x_shape[1], y_shape[1])), [x_shape, y_shape])]

	with tf.control_dependencies(assets):
		x = tf.identity(x)

	# N
	N = box_filter(tf.ones((1, 1, x_shape[2], x_shape[3]), dtype=x.dtype), r)

	# mean_x
	mean_x = box_filter(x, r) / N
	# mean_y
	mean_y = box_filter(y, r) / N
	# cov_xy
	cov_xy = box_filter(x * y, r) / N - mean_x * mean_y
	# var_x
	var_x = box_filter(x * x, r) / N - mean_x * mean_x

	# A
	A = cov_xy / (var_x + eps)
	# b
	b = mean_y - A * mean_x

	mean_A = box_filter(A, r) / N
	mean_b = box_filter(b, r) / N

	output = mean_A * x + mean_b

	if nhwc:
		output = tf.transpose(output, [0, 2, 3, 1])

	return output


_rgb_to_yuv_kernel = [[0.299, -0.14714119, 0.61497538],
                      [0.587, -0.28886916, -0.51496512],
                      [0.114, 0.43601035, -0.10001026]]


def rgb_to_yuv(images):
	"""Converts one or more images from RGB to YUV.
	Outputs a tensor of the same shape as the `images` tensor, containing the YUV
	value of the pixels.
	The output is only well defined if the value in images are in [0,1].
	Args:
	  images: 2-D or higher rank. Image data to convert. Last dimension must be
		size 3.
	Returns:
	  images: tensor with the same shape as `images`.
	"""
	images = ops.convert_to_tensor(images, name='images')
	kernel = ops.convert_to_tensor(
		_rgb_to_yuv_kernel, dtype=images.dtype, name='kernel')
	ndims = images.get_shape().ndims
	return math_ops.tensordot(images, kernel, axes=[[ndims - 1], [0]])


def bilinear_sampler_1d_h(input_images, x_offset, wrap_mode='border', name='bilinear_sampler', **kwargs):
	def _repeat(x, n_repeats):
		with tf.variable_scope('_repeat'):
			rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
			return tf.reshape(rep, [-1])

	def _interpolate(im, x, y):
		with tf.variable_scope('_interpolate'):

			# handle both texture border types
			_edge_size = 0
			if _wrap_mode == 'border':
				_edge_size = 1
				im = tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
				x = x + _edge_size
				y = y + _edge_size
			elif _wrap_mode == 'edge':
				_edge_size = 0
			else:
				return None

			x = tf.clip_by_value(x, 0.0, _width_f - 1 + 2 * _edge_size)

			x0_f = tf.floor(x)
			y0_f = tf.floor(y)
			x1_f = x0_f + 1

			x0 = tf.cast(x0_f, tf.int32)
			y0 = tf.cast(y0_f, tf.int32)
			x1 = tf.cast(tf.minimum(x1_f, _width_f - 1 + 2 * _edge_size), tf.int32)

			dim2 = (_width + 2 * _edge_size)
			dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)
			base = _repeat(tf.range(_num_batch) * dim1, _height * _width)
			base_y0 = base + y0 * dim2
			idx_l = base_y0 + x0
			idx_r = base_y0 + x1

			im_flat = tf.reshape(im, tf.stack([-1, _num_channels]))

			pix_l = tf.gather(im_flat, idx_l)
			pix_r = tf.gather(im_flat, idx_r)

			weight_l = tf.expand_dims(x1_f - x, 1)
			weight_r = tf.expand_dims(x - x0_f, 1)

			return weight_l * pix_l + weight_r * pix_r

	def _transform(input_images, x_offset):
		with tf.variable_scope('transform'):
			# grid of (x_t, y_t, 1), eq (1) in ref [1]
			x_t, y_t = tf.meshgrid(tf.linspace(0.0, _width_f - 1.0, _width),
			                       tf.linspace(0.0, _height_f - 1.0, _height))

			x_t_flat = tf.reshape(x_t, (1, -1))
			y_t_flat = tf.reshape(y_t, (1, -1))

			x_t_flat = tf.tile(x_t_flat, tf.stack([_num_batch, 1]))
			y_t_flat = tf.tile(y_t_flat, tf.stack([_num_batch, 1]))

			x_t_flat = tf.reshape(x_t_flat, [-1])
			y_t_flat = tf.reshape(y_t_flat, [-1])

			x_t_flat = x_t_flat + tf.reshape(x_offset, [-1]) * _width_f

			input_transformed = _interpolate(input_images, x_t_flat, y_t_flat)

			output = tf.reshape(
				input_transformed, tf.stack([_num_batch, _height, _width, _num_channels]))
			return output

	with tf.variable_scope(name):
		_num_batch = tf.shape(input_images)[0]
		_height = tf.shape(input_images)[1]
		_width = tf.shape(input_images)[2]
		_num_channels = tf.shape(input_images)[3]

		_height_f = tf.cast(_height, tf.float32)
		_width_f = tf.cast(_width, tf.float32)

		_wrap_mode = wrap_mode

		output = _transform(input_images, x_offset)
		return output


def upsample(x, scale=2, features=64, activation=tf.nn.relu):
	assert scale in [2, 3, 4]
	x = slim.conv2d(x, features, [3, 3], activation_fn=activation)
	if scale == 2:
		ps_features = (scale ** 2)
		x = slim.conv2d(x, ps_features, [3, 3], activation_fn=activation)
		x = PS(x, 2, color=False)
	elif scale == 3:
		ps_features = 3 * (scale ** 2)
		x = slim.conv2d(x, ps_features, [3, 3], activation_fn=activation)
		# x = slim.conv2d_transpose(x,ps_features,9,stride=1,activation_fn=activation)
		x = PS(x, 3, color=True)
	elif scale == 4:
		ps_features = 3 * (2 ** 2)
		for i in range(2):
			x = slim.conv2d(x, ps_features, [3, 3], activation_fn=activation)
			# x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
			x = PS(x, 2, color=True)
	return x


def resize_conv(x, scale, features=64, kernel_size=[3, 3], activation=None):
	''' Resize convolution : Upsampling using nearest neighbour followed by a convolution '''
	shape = x.shape
	# print("shape:",shape)
	new_h = shape[1] * scale
	new_w = shape[2] * scale
	# print("new shape: ",(new_h,new_w))
	x = tf.image.resize_images(x, (new_h, new_w), method=1, align_corners=True)
	# print('after bilinear: ',x.shape)
	x = slim.conv2d(x, features, kernel_size, activation_fn=activation)
	# print('after conv',x.shape)
	return x


"""
Borrowed from https://github.com/tetrachrome/subpixel
Used for subpixel phase shifting after deconv operations
"""


def _phase_shift(I, r):
	bsize, a, b, c = I.get_shape().as_list()
	bsize = tf.shape(I)[0]  # Handling Dimension(None) type for undefined batch dim
	X = tf.reshape(I, (bsize, a, b, r, r))
	X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
	X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
	X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, b, a*r, r
	X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
	X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, a*r, b*r
	return tf.reshape(X, (bsize, a * r, b * r, 1))


"""
Borrowed from https://github.com/tetrachrome/subpixel
Used for subpixel phase shifting after deconv operations
"""


def PS(X, r, color=False):
	if color:
		Xc = tf.split(X, 3, 3)
		X = tf.concat([_phase_shift(x, r) for x in Xc], 3)
	else:
		X = _phase_shift(X, r)
	return X


"""
Tensorflow log base 10.
Found here: https://github.com/tensorflow/tensorflow/issues/1666
"""


def log10(x):
	numerator = tf.log(x)
	denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
	return numerator / denominator
