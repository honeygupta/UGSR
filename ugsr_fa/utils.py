"""
Project: Unaligned Guided Thermal Image Super-Resolution
Author: Honey Gupta (hn.gpt1@gmail.com)

"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

_SSIM_K1 = 0.01
_SSIM_K2 = 0.03

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


_MSSSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def convert_image_dtype(image, dtype, saturate=False, name=None):
	"""Convert `image` to `dtype`, scaling its values if needed.
	Images that are represented using floating point values are expected to have
	values in the range [0,1). Image data stored in integer data types are
	expected to have values in the range `[0,MAX]`, where `MAX` is the largest
	positive representable number for the data type.
	This op converts between data types, scaling the values appropriately before
	casting.
	Note that converting from floating point inputs to integer types may lead to
	over/underflow problems. Set saturate to `True` to avoid such problem in
	problematic conversions. If enabled, saturation will clip the output into the
	allowed range before performing a potentially dangerous cast (and only before
	performing such a cast, i.e., when casting from a floating point to an integer
	type, and when casting from a signed to an unsigned type; `saturate` has no
	effect on casts between floats, or on casts that increase the type's range).
	Args:
	  image: An image.
	  dtype: A `DType` to convert `image` to.
	  saturate: If `True`, clip the input before casting (if necessary).
	  name: A name for this operation (optional).
	Returns:
	  `image`, converted to `dtype`.
	"""
	image = ops.convert_to_tensor(image, name='image')
	if dtype == image.dtype:
		return array_ops.identity(image, name=name)

	with ops.name_scope(name, 'convert_image', [image]) as name:
		# Both integer: use integer multiplication in the larger range
		if image.dtype.is_integer and dtype.is_integer:
			scale_in = image.dtype.max
			scale_out = dtype.max
			if scale_in > scale_out:
				# Scaling down, scale first, then cast. The scaling factor will
				# cause in.max to be mapped to above out.max but below out.max+1,
				# so that the output is safely in the supported range.
				scale = (scale_in + 1) // (scale_out + 1)
				scaled = math_ops.div(image, scale)

				if saturate:
					return math_ops.saturate_cast(scaled, dtype, name=name)
				else:
					return math_ops.cast(scaled, dtype, name=name)
			else:
				# Scaling up, cast first, then scale. The scale will not map in.max to
				# out.max, but converting back and forth should result in no change.
				if saturate:
					cast = math_ops.saturate_cast(image, dtype)
				else:
					cast = math_ops.cast(image, dtype)
				scale = (scale_out + 1) // (scale_in + 1)
				return math_ops.multiply(cast, scale, name=name)
		elif image.dtype.is_floating and dtype.is_floating:
			# Both float: Just cast, no possible overflows in the allowed ranges.
			# Note: We're ignoreing float overflows. If your image dynamic range
			# exceeds float range you're on your own.
			return math_ops.cast(image, dtype, name=name)
		else:
			if image.dtype.is_integer:
				# Converting to float: first cast, then scale. No saturation possible.
				cast = math_ops.cast(image, dtype)
				scale = 1. / image.dtype.max
				return math_ops.multiply(cast, scale, name=name)
			else:
				# Converting from float: first scale, then cast
				scale = dtype.max + 0.5  # avoid rounding problems in the cast
				scaled = math_ops.multiply(image, scale)
				if saturate:
					return math_ops.saturate_cast(scaled, dtype, name=name)
				else:
					return math_ops.cast(scaled, dtype, name=name)


def _ssim_helper(x, y, reducer, max_val, compensation=1.0):
	c1 = (_SSIM_K1 * max_val) ** 2
	c2 = (_SSIM_K2 * max_val) ** 2

	# SSIM luminance measure is
	# (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1).
	mean0 = reducer(x)
	mean1 = reducer(y)
	num0 = mean0 * mean1 * 2.0
	den0 = math_ops.square(mean0) + math_ops.square(mean1)
	luminance = (num0 + c1) / (den0 + c1)

	# SSIM contrast-structure measure is
	#   (2 * cov_{xy} + c2) / (cov_{xx} + cov_{yy} + c2).
	# Note that `reducer` is a weighted sum with weight w_k, \sum_i w_i = 1, then
	#   cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
	#          = \sum_i w_i x_i y_i - (\sum_i w_i x_i) (\sum_j w_j y_j).
	num1 = reducer(x * y) * 2.0
	den1 = reducer(math_ops.square(x) + math_ops.square(y))
	c2 *= compensation
	cs = (num1 - num0 + c2) / (den1 - den0 + c2)

	# SSIM score is the product of the luminance and contrast-structure measures.
	return luminance, cs


def _fspecial_gauss(size, sigma):
	"""Function to mimic the 'fspecial' gaussian MATLAB function."""
	size = ops.convert_to_tensor(size, dtypes.int32)
	sigma = ops.convert_to_tensor(sigma)

	coords = math_ops.cast(math_ops.range(size), sigma.dtype)
	coords -= math_ops.cast(size - 1, sigma.dtype) / 2.0

	g = math_ops.square(coords)
	g *= -0.5 / math_ops.square(sigma)

	g = array_ops.reshape(g, shape=[1, -1]) + array_ops.reshape(g, shape=[-1, 1])
	g = array_ops.reshape(g, shape=[1, -1])  # For tf.nn.softmax().
	g = nn_ops.softmax(g)
	return array_ops.reshape(g, shape=[size, size, 1, 1])


def _ssim_per_channel(img1, img2, max_val=1.0):
	""""""
	filter_size = constant_op.constant(11, dtype=dtypes.int32)
	filter_sigma = constant_op.constant(1.5, dtype=img1.dtype)

	shape1, shape2 = array_ops.shape_n([img1, img2])
	checks = [
		control_flow_ops.Assert(math_ops.reduce_all(math_ops.greater_equal(
			shape1[-3:-1], filter_size)), [shape1, filter_size], summarize=8),
		control_flow_ops.Assert(math_ops.reduce_all(math_ops.greater_equal(
			shape2[-3:-1], filter_size)), [shape2, filter_size], summarize=8)]

	# Enforce the check to run before computation.
	with ops.control_dependencies(checks):
		img1 = array_ops.identity(img1)

	# TODO(sjhwang): Try to cache kernels and compensation factor.
	kernel = _fspecial_gauss(filter_size, filter_sigma)
	kernel = array_ops.tile(kernel, multiples=[1, 1, shape1[-1], 1])

	# The correct compensation factor is `1.0 - tf.reduce_sum(tf.square(kernel))`,
	# but to match MATLAB implementation of MS-SSIM, we use 1.0 instead.
	compensation = 1.0

	# TODO(sjhwang): Try FFT.
	# TODO(sjhwang): Gaussian kernel is separable in space. Consider applying
	#   1-by-n and n-by-1 Gaussain filters instead of an n-by-n filter.
	def reducer(x):
		shape = array_ops.shape(x)
		x = array_ops.reshape(x, shape=array_ops.concat([[-1], shape[-3:]], 0))
		y = nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
		return array_ops.reshape(y, array_ops.concat([shape[:-3],
		                                              array_ops.shape(y)[1:]], 0))

	luminance, cs = _ssim_helper(img1, img2, reducer, max_val, compensation)

	# Average over the second and the third from the last: height, width.
	axes = constant_op.constant([-3, -2], dtype=dtypes.int32)
	ssim_val = math_ops.reduce_mean(luminance * cs, axes)
	cs = math_ops.reduce_mean(cs, axes)
	return ssim_val, cs


def ssim_multiscale(img1, img2, max_val, power_factors=_MSSSIM_WEIGHTS):
	# Shape checking.
	shape1 = img1.get_shape().with_rank_at_least(3)
	shape2 = img2.get_shape().with_rank_at_least(3)
	shape1[-3:].merge_with(shape2[-3:])

	with ops.name_scope(None, 'MS-SSIM', [img1, img2]):
		shape1, shape2, checks = _verify_compatible_image_shapes(img1, img2)
		with ops.control_dependencies(checks):
			img1 = array_ops.identity(img1)

		# Need to convert the images to float32.  Scale max_val accordingly so that
		# SSIM is computed correctly.
		max_val = math_ops.cast(max_val, img1.dtype)
		max_val = convert_image_dtype(max_val, dtypes.float32)
		img1 = convert_image_dtype(img1, dtypes.float32)
		img2 = convert_image_dtype(img2, dtypes.float32)

		imgs = [img1, img2]
		shapes = [shape1, shape2]

		# img1 and img2 are assumed to be a (multi-dimensional) batch of
		# 3-dimensional images (height, width, channels). `heads` contain the batch
		# dimensions, and `tails` contain the image dimensions.
		heads = [s[:-3] for s in shapes]
		tails = [s[-3:] for s in shapes]

		divisor = [1, 2, 2, 1]
		divisor_tensor = constant_op.constant(divisor[1:], dtype=dtypes.int32)

		def do_pad(images, remainder):
			padding = array_ops.expand_dims(remainder, -1)
			padding = array_ops.pad(padding, [[1, 0], [1, 0]])
			return [array_ops.pad(x, padding, mode='SYMMETRIC') for x in images]

		mcs = []
		for k in range(len(power_factors)):
			with ops.name_scope(None, 'Scale%d' % k, imgs):
				if k > 0:
					# Avg pool takes rank 4 tensors. Flatten leading dimensions.
					flat_imgs = [
						array_ops.reshape(x, array_ops.concat([[-1], t], 0))
						for x, t in zip(imgs, tails)
					]

					remainder = tails[0] % divisor_tensor
					need_padding = math_ops.reduce_any(math_ops.not_equal(remainder, 0))
					# pylint: disable=cell-var-from-loop
					padded = control_flow_ops.cond(need_padding,
					                               lambda: do_pad(flat_imgs, remainder),
					                               lambda: flat_imgs)
					# pylint: enable=cell-var-from-loop

					downscaled = [nn_ops.avg_pool(x, ksize=divisor, strides=divisor,
					                              padding='VALID')
					              for x in padded]
					tails = [x[1:] for x in array_ops.shape_n(downscaled)]
					imgs = [
						array_ops.reshape(x, array_ops.concat([h, t], 0))
						for x, h, t in zip(downscaled, heads, tails)
					]

				# Overwrite previous ssim value since we only need the last one.
				ssim_per_channel, cs = _ssim_per_channel(*imgs, max_val=max_val)
				mcs.append(nn_ops.relu(cs))

		# Remove the cs score for the last scale. In the MS-SSIM calculation,
		# we use the l(p) at the highest scale. l(p) * cs(p) is ssim(p).
		mcs.pop()  # Remove the cs score for the last scale.
		mcs_and_ssim = array_ops.stack(mcs + [nn_ops.relu(ssim_per_channel)],
		                               axis=-1)
		# Take weighted geometric mean across the scale axis.
		ms_ssim = math_ops.reduce_prod(math_ops.pow(mcs_and_ssim, power_factors),
		                               [-1])

	return math_ops.reduce_mean(ms_ssim, [-1])


def _verify_compatible_image_shapes(img1, img2):
	"""Checks if two image tensors are compatible for applying SSIM or PSNR.
	This function checks if two sets of images have ranks at least 3, and if the
	last three dimensions match.
	Args:
	  img1: Tensor containing the first image batch.
	  img2: Tensor containing the second image batch.
	Returns:
	  A tuple containing: the first tensor shape, the second tensor shape, and a
	  list of control_flow_ops.Assert() ops implementing the checks.
	Raises:
	  ValueError: When static shape check fails.
	"""
	shape1 = img1.get_shape().with_rank_at_least(3)
	shape2 = img2.get_shape().with_rank_at_least(3)
	shape1[-3:].assert_is_compatible_with(shape2[-3:])

	if shape1.ndims is not None and shape2.ndims is not None:
		for dim1, dim2 in zip(reversed(shape1[:-3]), reversed(shape2[:-3])):
			if not (dim1 == 1 or dim2 == 1 or dim1.is_compatible_with(dim2)):
				raise ValueError(
					'Two images are not compatible: %s and %s' % (shape1, shape2))

	# Now assign shape tensors.
	shape1, shape2 = array_ops.shape_n([img1, img2])

	# TODO(sjhwang): Check if shape1[:-3] and shape2[:-3] are broadcastable.
	checks = []
	checks.append(control_flow_ops.Assert(
		math_ops.greater_equal(array_ops.size(shape1), 3),
		[shape1, shape2], summarize=10))
	checks.append(control_flow_ops.Assert(
		math_ops.reduce_all(math_ops.equal(shape1[-3:], shape2[-3:])),
		[shape1, shape2], summarize=10))
	return shape1, shape2, checks


def psnr(a, b, max_val, name=None):
	"""Returns the Peak Signal-to-Noise Ratio between a and b.
	This is intended to be used on signals (or images). Produces a PSNR value for
	each image in batch.
	The last three dimensions of input are expected to be [height, width, depth].
	Example:
	```python
		# Read images from file.
		im1 = tf.decode_png('path/to/im1.png')
		im2 = tf.decode_png('path/to/im2.png')
		# Compute PSNR over tf.uint8 Tensors.
		psnr1 = tf.image.psnr(im1, im2, max_val=255)
		# Compute PSNR over tf.float32 Tensors.
		im1 = tf.image.convert_image_dtype(im1, tf.float32)
		im2 = tf.image.convert_image_dtype(im2, tf.float32)
		psnr2 = tf.image.psnr(im1, im2, max_val=1.0)
		# psnr1 and psnr2 both have type tf.float32 and are almost equal.
	```
	Arguments:
	  a: First set of images.
	  b: Second set of images.
	  max_val: The dynamic range of the images (i.e., the difference between the
		maximum the and minimum allowed values).
	  name: Namespace to embed the computation in.
	Returns:
	  The scalar PSNR between a and b. The returned tensor has type `tf.float32`
	  and shape [batch_size, 1].
	"""
	with ops.name_scope(name, 'PSNR', [a, b]):
		# Need to convert the images to float32.  Scale max_val accordingly so that
		# PSNR is computed correctly.
		max_val = math_ops.cast(max_val, a.dtype)
		max_val = convert_image_dtype(max_val, dtypes.float32)
		a = convert_image_dtype(a, dtypes.float32)
		b = convert_image_dtype(b, dtypes.float32)
		mse = math_ops.reduce_mean(math_ops.squared_difference(a, b), [-3, -2, -1])
		psnr_val = math_ops.subtract(
			20 * math_ops.log(max_val) / math_ops.log(10.0),
			np.float32(10 / np.log(10)) * math_ops.log(mse),
			name='psnr')

		_, _, checks = _verify_compatible_image_shapes(a, b)
		with ops.control_dependencies(checks):
			return array_ops.identity(psnr_val)


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
