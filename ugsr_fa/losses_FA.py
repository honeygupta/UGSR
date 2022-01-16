"""
Project: Unaligned Guided Thermal Image Super-Resolution
Author: Honey Gupta (hn.gpt1@gmail.com)

"""
from enum import Enum
from logging import exception

import numpy as np
import tensorflow as tf

import model
import prefer_static


def assert_is_compatible_with(x, other):
	tf.TensorShape(x).assert_is_compatible_with(other)


def variance(x, sample_axis=0, keep_dims=False, name=None):
	with tf.name_scope(name or 'variance'):
		return covariance(
			x, y=None, sample_axis=sample_axis, event_axis=None, keep_dims=keep_dims)


def stddev(x, sample_axis=0, keep_dims=False, name=None):
	with tf.name_scope(name or 'stddev'):
		return tf.sqrt(variance(x, sample_axis=sample_axis, keep_dims=keep_dims))


def _is_list_like(x):
	"""Helper which returns `True` if input is `list`-like."""
	return isinstance(x, (tuple, list))


def rank(x):
	return tf.rank


def _make_list_or_1d_tensor(values):
	"""Return a list (preferred) or 1d Tensor from values, if values.ndims < 2."""
	values = tf.convert_to_tensor(values, name='values')
	values_ = tf.contrib.util.constant_value(values)

	# Static didn't work.
	if values_ is None:
		# Cheap way to bring to at least 1d.
		return values + tf.zeros([1], dtype=values.dtype)

	# Static worked!
	if values_.ndim > 1:
		raise ValueError('values had > 1 dim: {}'.format(values_.shape))
	# Cheap way to bring to at least 1d.
	values_ = values_ + np.zeros([1], dtype=values_.dtype)
	return list(values_)


def _make_positive_axis(axis, ndims):
	"""Rectify possibly negatively axis. Prefer return Python list."""
	axis = _make_list_or_1d_tensor(axis)

	ndims = tf.convert_to_tensor(ndims, name='ndims', dtype=tf.int32)
	ndims_ = tf.contrib.util.constant_value(ndims)

	if _is_list_like(axis) and ndims_ is not None:
		# Static case
		positive_axis = []
		for a in axis:
			if a < 0:
				a = ndims_ + a
			positive_axis.append(a)
	else:
		# Dynamic case
		axis = tf.convert_to_tensor(axis, name='axis', dtype=tf.int32)
		positive_axis = tf.where(axis >= 0, axis, axis + ndims)

	return positive_axis


def covariance(x,
               y=None,
               sample_axis=0,
               event_axis=-1,
               keep_dims=False,
               name=None):
	with tf.name_scope(name or 'covariance'):
		x = tf.convert_to_tensor(x, name='x')
		# Covariance *only* uses the centered versions of x (and y).
		x -= tf.reduce_mean(x, axis=sample_axis, keep_dims=True)

		if y is None:
			y = x
		else:
			y = tf.convert_to_tensor(y, name='y', dtype=x.dtype)
			# If x and y have different shape, sample_axis and event_axis will likely
			# be wrong for one of them!
			assert_is_compatible_with(x.shape, y.shape)
			y -= tf.reduce_mean(y, axis=sample_axis, keep_dims=True)

		if event_axis is None:
			return tf.reduce_mean(
				x * tf.conj(y), axis=sample_axis, keep_dims=keep_dims)

		if sample_axis is None:
			raise ValueError(
				'sample_axis was None, which means all axis hold events, and this '
				'overlaps with event_axis ({})'.format(event_axis))

		event_axis = _make_positive_axis(event_axis, tf.rank(x))
		sample_axis = _make_positive_axis(sample_axis, tf.rank(x))

		# If we get lucky and axis is statically defined, we can do some checks.
		if _is_list_like(event_axis) and _is_list_like(sample_axis):
			if set(event_axis).intersection(sample_axis):
				raise ValueError(
					'sample_axis ({}) and event_axis ({}) overlapped'.format(
						sample_axis, event_axis))
			if (np.diff(sorted(event_axis)) > 1).any():
				raise ValueError(
					'event_axis must be contiguous. Found: {}'.format(event_axis))
			batch_axis = []  # list(sorted(set(range(rank(x.shape))).difference(sample_axis + event_axis)))
		else:
			batch_axis = prefer_static.setdiff1d(
				tf.range(0, tf.rank(x)), tf.concat((sample_axis, event_axis), 0))

		event_axis = tf.convert_to_tensor(
			event_axis, name='event_axis', dtype=tf.int32)
		sample_axis = tf.convert_to_tensor(
			sample_axis, name='sample_axis', dtype=tf.int32)
		batch_axis = tf.convert_to_tensor(
			batch_axis, name='batch_axis', dtype=tf.int32)

		# Permute x/y until shape = B + E + S
		perm_for_xy = tf.concat((batch_axis, event_axis, sample_axis), 0)
		x_permed = tf.transpose(a=x, perm=perm_for_xy)
		y_permed = tf.transpose(a=y, perm=perm_for_xy)

		batch_ndims = tf.size(batch_axis)
		batch_shape = tf.shape(x_permed)[:batch_ndims]
		event_ndims = tf.size(event_axis)
		event_shape = tf.shape(x_permed)[batch_ndims:batch_ndims + event_ndims]
		sample_shape = tf.shape(x_permed)[batch_ndims + event_ndims:]
		sample_ndims = tf.size(sample_shape)
		n_samples = tf.reduce_prod(sample_shape)
		n_events = tf.reduce_prod(event_shape)

		# Flatten sample_axis into one long dim.
		x_permed_flat = tf.reshape(
			x_permed, tf.concat((batch_shape, event_shape, [n_samples]), 0))
		y_permed_flat = tf.reshape(
			y_permed, tf.concat((batch_shape, event_shape, [n_samples]), 0))
		# Do the same for event_axis.
		x_permed_flat = tf.reshape(
			x_permed, tf.concat((batch_shape, [n_events], [n_samples]), 0))
		y_permed_flat = tf.reshape(
			y_permed, tf.concat((batch_shape, [n_events], [n_samples]), 0))

		# After matmul, cov.shape = batch_shape + [n_events, n_events]
		cov = tf.matmul(
			x_permed_flat, y_permed_flat, adjoint_b=True) / tf.cast(
			n_samples, x.dtype)

		cov = tf.reshape(
			cov,
			tf.concat(
				(
					batch_shape,
					# event_shape**2 used here because it is the same length as
					# event_shape, and has the same number of elements as one
					# batch of covariance.
					event_shape ** 2,
					tf.ones([sample_ndims], tf.int32)),
				0))

		cov = tf.transpose(a=cov, perm=tf.invert_permutation(perm_for_xy))

		e_start = event_axis[0]
		e_len = 1 + event_axis[-1] - event_axis[0]
		cov = tf.reshape(
			cov,
			tf.concat((tf.shape(cov)[:e_start], event_shape, event_shape,
			           tf.shape(cov)[e_start + e_len:]), 0))

		# tf.squeeze requires python ints for axis, not Tensor.  This is enough to
		# require our axis args to be constants.
		if not keep_dims:
			squeeze_axis = tf.where(sample_axis < e_start, sample_axis,
			                        sample_axis + e_len)
			cov = tf.squeeze(cov)

		return cov


def correlation(x,
                y=None,
                sample_axis=0,
                event_axis=-1,
                keep_dims=False,
                name=None):
	with tf.name_scope(name or 'correlation'):
		x /= stddev(x, sample_axis=sample_axis, keep_dims=True)
		if y is not None:
			y /= stddev(y, sample_axis=sample_axis, keep_dims=True)

		return covariance(
			x=x,
			y=y,
			event_axis=event_axis,
			sample_axis=sample_axis,
			keep_dims=keep_dims)


def gradient_sparcity(image):
	return tf.reduce_mean(image_gradients(image))


def fa_corrLoss(features, epsilon=1e-3):
	guide = features[0]
	ther = features[1]
	l1 = tf.reshape(guide, [model.BATCH_SIZE, 64 * 80, -1])
	l2 = tf.reshape(ther, [model.BATCH_SIZE, 64 * 80, -1])
	corr_mat_t = []
	for i in range(model.BATCH_SIZE):
		corr_loss = correlation(l1[i, :], l2[i, :], sample_axis=0, event_axis=-1, keep_dims=False)
		corr_mat_t.append(corr_loss)

	# I = tf.eye(tf.shape(guide)[-1],batch_shape=[model.BATCH_SIZE])
	return (-tf.log(tf.reduce_mean(tf.matrix_diag(corr_mat_t))))


def faLoss(features):
	# print(features)
	guide = features[0]
	ther = features[1]
	ther, guide = center_by_ther(ther, guide)
	ther = l2_normalize_channelwise(ther)
	guide = l2_normalize_channelwise(guide)
	return tf.losses.cosine_distance(ther, guide, 2)


def center_by_ther(T_features, G_features):
	axes = [0, 1, 2]
	meanT, varT = tf.nn.moments(T_features, axes)
	T_features_centered = T_features - meanT
	G_features_centered = G_features - meanT

	return T_features_centered, G_features_centered


def l2_normalize_channelwise(features):
	norms = tf.norm(features, ord='euclidean', axis=TensorAxis.C, name='norm')
	norms_expanded = tf.expand_dims(norms, TensorAxis.C)
	features = tf.divide(features, norms_expanded, name='normalized')
	return features


def lsgan_loss_generator(prob_fake_is_real):
	return tf.reduce_mean(tf.squared_difference(prob_fake_is_real, 1))


def lsgan_loss_discriminator(prob_real_is_real, prob_fake_is_real):
	return (tf.reduce_mean(tf.squared_difference(prob_real_is_real, 1)) +
	        tf.reduce_mean(tf.squared_difference(prob_fake_is_real, 0))) * 0.5


def rec_loss(output, gt):
	return tf.losses.absolute_difference(gt, output)


from tensorflow.python.ops import array_ops


def image_gradients(image):
	if image.get_shape().ndims != 4:
		raise ValueError('image_gradients expects a 4D tensor '
		                 '[batch_size, h, w, d], not %s.', image.get_shape())

	image_shape = array_ops.shape(image)
	batch_size, height, width, depth = array_ops.unstack(image_shape)
	dy = image[:, 1:, :, :] - image[:, :-1, :, :]
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


class Distance(Enum):
	L2 = 0
	DotProduct = 1


class TensorAxis:
	N = 0
	H = 1
	W = 2
	C = 3


class CSFlow:
	def __init__(self, sigma=float(0.1), b=float(1.0)):
		self.b = b
		self.sigma = sigma

	def __calculate_CS(self, scaled_distances, axis_for_normalization=TensorAxis.C):
		self.scaled_distances = scaled_distances
		self.cs_weights_before_normalization = tf.exp((self.b - scaled_distances) / self.sigma,
		                                              name='weights_before_normalization')
		self.cs_NHWC = CSFlow.sum_normalize(self.cs_weights_before_normalization, axis_for_normalization)

	def reversed_direction_CS(self):
		cs_flow_opposite = CSFlow(self.sigma, self.b)
		cs_flow_opposite.raw_distances = self.raw_distances
		work_axis = [TensorAxis.H, TensorAxis.W]
		relative_dist = cs_flow_opposite.calc_relative_distances(axis=work_axis)
		cs_flow_opposite.__calculate_CS(relative_dist, work_axis)
		return cs_flow_opposite

	@staticmethod
	def create_using_dotP(I_features, T_features, sigma=float(1.0), b=float(1.0)):
		cs_flow = CSFlow(sigma, b)
		with tf.name_scope('CS'):
			# T_features, I_features = cs_flow.center_by_T(T_features, I_features)
			with tf.name_scope('TFeatures'):
				T_features = CSFlow.l2_normalize_channelwise(T_features)
			with tf.name_scope('IFeatures'):
				I_features = CSFlow.l2_normalize_channelwise(I_features)

				# work seperatly for each example in dim 1
				cosine_dist_l = []
				N, _, __, ___ = T_features.shape.as_list()
				for i in range(N):
					T_features_i = tf.expand_dims(T_features[i, :, :, :], 0)
					I_features_i = tf.expand_dims(I_features[i, :, :, :], 0)
					patches_HWCN_i = cs_flow.patch_decomposition(T_features_i)
					cosine_dist_i = tf.nn.conv2d(I_features_i, patches_HWCN_i, strides=[1, 1, 1, 1],
					                             padding='VALID', use_cudnn_on_gpu=True, name='cosine_dist')
					cosine_dist_l.append(cosine_dist_i)

				cs_flow.cosine_dist = tf.concat(cosine_dist_l, axis=0)

				cosine_dist_zero_to_one = -(cs_flow.cosine_dist - 1) / 2
				cs_flow.raw_distances = cosine_dist_zero_to_one

				relative_dist = cs_flow.calc_relative_distances()
				# cs_flow.__calculate_CS(relative_dist)
				return relative_dist

	def calc_relative_distances(self, axis=TensorAxis.C):
		epsilon = 1e-5
		div = tf.reduce_min(self.raw_distances, axis=axis, keep_dims=True)
		# div = tf.reduce_mean(self.raw_distances, axis=axis, keep_dims=True)
		relative_dist = self.raw_distances / (div + epsilon)
		return relative_dist

	def weighted_average_dist(self, axis=TensorAxis.C):
		if not hasattr(self, 'raw_distances'):
			raise exception('raw_distances property does not exists. cant calculate weighted average l2')

		multiply = self.raw_distances * self.cs_NHWC
		return tf.reduce_sum(multiply, axis=axis, name='weightedDistPerPatch')

	# --
	@staticmethod
	def create(I_features, T_features, nnsigma=float(1.0), b=float(1.0)):
		cs_flow = CSFlow.create_using_dotP(I_features, T_features, nnsigma, b)
		return cs_flow

	@staticmethod
	def sum_normalize(cs, axis=TensorAxis.C):
		reduce_sum = tf.reduce_sum(cs, axis, keep_dims=True, name='sum')
		return tf.divide(cs, reduce_sum, name='sumNormalized')

	def center_by_T(self, T_features, I_features):
		# assuming both input are of the same size

		# calculate stas over [batch, height, width], expecting 1x1xDepth tensor
		axes = [0, 1, 2]
		self.meanT, self.varT = tf.nn.moments(
			T_features, axes, name='TFeatures/moments')

		# we do not divide by std since its causing the histogram
		# for the final cs to be very thin, so the NN weights
		# are not distinctive, giving similar values for all patches.
		# stdT = tf.sqrt(varT, "stdT")
		# correct places with std zero
		# stdT[tf.less(stdT, tf.constant(0.001))] = tf.constant(1)

		# TODO check broadcasting here
		with tf.name_scope('TFeatures/centering'):
			self.T_features_centered = T_features - self.meanT
		with tf.name_scope('IFeatures/centering'):
			self.I_features_centered = I_features - self.meanT

		return self.T_features_centered, self.I_features_centered

	@staticmethod
	def l2_normalize_channelwise(features):
		norms = tf.norm(features, ord='euclidean', axis=TensorAxis.C, name='norm')
		# expanding the norms tensor to support broadcast division
		norms_expanded = tf.expand_dims(norms, TensorAxis.C)
		features = tf.divide(features, norms_expanded, name='normalized')
		return features

	def patch_decomposition(self, T_features):
		# patch decomposition
		# see https://stackoverflow.com/questions/40731433/understanding-tf-extract-image-patches-for-extracting-patches-from-an-image
		patch_size = 1
		patches_as_depth_vectors = tf.extract_image_patches(
			images=T_features, ksizes=[1, patch_size, patch_size, 1],
			strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID',
			name='patches_as_depth_vectors')

		self.patches_NHWC = tf.reshape(
			patches_as_depth_vectors,
			shape=[-1, patch_size, patch_size, patches_as_depth_vectors.shape[3].value],
			name='patches_PHWC')

		self.patches_HWCN = tf.transpose(
			self.patches_NHWC,
			perm=[1, 2, 3, 0],
			name='patches_HWCP')  # tf.conv2 ready format

		return self.patches_HWCN


def contextual_loss(T_features, I_features):
	with tf.name_scope('FA'):
		cs_flow = CSFlow.create(I_features, T_features)
		CS = tf.reduce_mean(cs_flow, axis=[1])
		CX_as_loss = 1 - CS
		CX_loss = -tf.log(1 - CX_as_loss)
		CX_loss = tf.reduce_mean(CX_loss)
		return CX_loss
