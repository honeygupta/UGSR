"""
Project: Unaligned Guided Thermal Image Super-Resolution
Author: Honey Gupta (hn.gpt1@gmail.com)

"""

import tensorflow as tf

import layers

slim = tf.contrib.slim
import ops

BATCH_SIZE = 1

H_IMG_HEIGHT = 512  # 512 for flir and 480 for cats
H_IMG_WIDTH = 640

G_IMG_HEIGHT = H_IMG_HEIGHT
G_IMG_WIDTH = H_IMG_WIDTH

L_IMG_HEIGHT = H_IMG_HEIGHT // 8
L_IMG_WIDTH = H_IMG_WIDTH // 8

IMG_CHANNELS = 3

ngf = 32
ndf = 64


def get_outputs(inputs):
	images_in = inputs['image_in']
	images_gd = inputs['image_gd']

	with tf.variable_scope("Model") as scope:
		current_generator = generator

		features, output = current_generator(images_in, images_gd, name="g_")

		return {
			'output': output,
			'features': features,
		}


def TransitionDown(inputs, n_filters, scope=None):
	with tf.name_scope(scope) as sc:
		preact = tf.nn.relu(inputs)
		conv = slim.conv2d(preact, n_filters, stride=[2, 2], kernel_size=3, padding='SAME', activation_fn=None,
		                   normalizer_fn=None,
		                   weights_initializer=tf.truncated_normal_initializer(
			                   stddev=0.02), biases_initializer=tf.constant_initializer(0.0))

		return conv


def preact_conv(inputs, n_filters, kernel_size=[3, 3]):
	preact = tf.nn.relu(inputs)
	conv = slim.conv2d(preact, n_filters, kernel_size, activation_fn=None, normalizer_fn=None,
	                   weights_initializer=tf.truncated_normal_initializer(
		                   stddev=0.02), biases_initializer=tf.constant_initializer(0.0))
	return conv


def TransitionUp(block_to_upsample, skip_connection, n_filters_keep, scope=None):
	with tf.name_scope(scope) as sc:
		l = slim.conv2d_transpose(block_to_upsample, n_filters_keep, kernel_size=[3, 3], stride=[2, 2],
		                          activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
		                          biases_initializer=tf.constant_initializer(0.0))
		l = tf.concat([l, skip_connection], axis=-1)
		return l


def DenseBlock(stack, n_layers, growth_rate, scope=None):
	with tf.name_scope(scope) as sc:
		new_features = []
		for j in range(n_layers):
			layer = preact_conv(stack, growth_rate)
			new_features.append(layer)
			stack = tf.concat([stack, layer], axis=-1)
		new_features = tf.concat(new_features, axis=-1)
		return stack, new_features


def generator(inputA, guide, name="generator"):
	f = 7
	ks = 3

	n_pool = 3
	growth_rate = 32
	n_layers_per_block = [2, 2, 2, 2, 2, 2, 2]

	with tf.variable_scope(name):

		if type(n_layers_per_block) == list:
			assert (len(n_layers_per_block) == 2 * n_pool + 1)
		elif type(n_layers_per_block) == int:
			n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
		else:
			raise ValueError

		stack1 = layers.general_conv2d(guide, ngf, 3, 3, 1, 1, 0.02, "SAME", "c0")
		n_filters = ngf

		for i in range(n_pool):
			stack1, _ = DenseBlock(stack1, n_layers_per_block[i], growth_rate, scope='denseblock%d' % (i + 1))
			n_filters += growth_rate * n_layers_per_block[i]
			stack1 = TransitionDown(stack1, n_filters, scope='transitiondown%d' % (i + 1))
		# print('Guide ', stack1)

		# Thermal branch

		skip_connection_list = []
		n_filters = ngf
		o_c1 = layers.general_conv2d(
			inputA, ngf, f, f, 1, 1, 0.02, name="c1", padding="SAME")
		o_c2 = layers.general_conv2d(
			o_c1, ngf * 2, ks, ks, 1, 1, 0.02, "SAME", "c2")
		o_c3 = layers.general_conv2d(
			o_c2, ngf * 2, ks, ks, 1, 1, 0.02, "SAME", "c3")

		stack2 = slim.conv2d(o_c3, ngf * 2, [3, 3], padding='SAME', activation_fn=tf.nn.leaky_relu,
		                     weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
		                                                                                        mode='FAN_IN',
		                                                                                        uniform=False))

		for i in range(n_pool):
			stack2, _ = DenseBlock(stack2, n_layers_per_block[i], growth_rate, scope='denseblock%d' % (i + 1))
			n_filters += growth_rate * n_layers_per_block[i]
			skip_connection_list.append(stack2)
			stack2 = preact_conv(stack2, n_filters, kernel_size=[1, 1])

		print(skip_connection_list)

		# print(skip_connection_list)
		skip_connection_list[0] = tf.image.resize_images(skip_connection_list[0],
		                                                 tf.shape(skip_connection_list[0])[1:3] * 8, align_corners=True)
		skip_connection_list[1] = tf.image.resize_images(skip_connection_list[1],
		                                                 tf.shape(skip_connection_list[1])[1:3] * 4, align_corners=True)
		skip_connection_list[2] = tf.image.resize_images(skip_connection_list[2],
		                                                 tf.shape(skip_connection_list[2])[1:3] * 2, align_corners=True)
		# print(skip_connection_list)

		skip_connection_list = skip_connection_list[::-1]

		# stack = tf.concat([stack1,stack2],axis=3)
		stack1_befAt = tf.nn.tanh(stack1)
		stack2_befAt = tf.nn.tanh(stack2)

		stack1_affAt = google_attention(x=stack1_befAt, channels=n_filters, scope='attention_guide')
		stack2_affAt = google_attention(x=stack2_befAt, channels=n_filters, scope='attention_ther')

		print('stack1_affAt', stack1_affAt)
		print('stack2_affAt', stack2_affAt)

		stack = (stack1_affAt + stack2_affAt) / 2.0

		stack, block_to_upsample = DenseBlock(stack, n_layers_per_block[n_pool], growth_rate,
		                                      scope='denseblock%d' % (n_pool + 1))

		for i in range(n_pool):
			n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
			stack = TransitionUp(block_to_upsample, skip_connection_list[i], n_filters_keep,
			                     scope='transitionup%d' % (n_pool + i + 1))
			stack, block_to_upsample = DenseBlock(stack, n_layers_per_block[n_pool + i + 1], growth_rate,
			                                      scope='denseblock%d' % (n_pool + i + 2))

		net = tf.nn.tanh(
			layers.general_conv2d(stack, 1, 3, 3, 1, 1, 0.02, "SAME", "c6", do_norm=False, do_relu=False))

		net = (net + tf.image.resize_images(inputA, [H_IMG_HEIGHT, H_IMG_WIDTH])) / 2.0

		return [stack1, stack2], net


def google_attention(x, channels, scope='attention'):
	sn = True
	with tf.variable_scope(scope):
		batch_size, height, width, num_channels = x.get_shape().as_list()
		f = ops.conv(x, channels // 8, kernel=1, stride=1, sn=sn, scope='f_conv')  # [bs, h, w, c']
		f = ops.max_pooling(f)

		g = ops.conv(x, channels // 8, kernel=1, stride=1, sn=sn, scope='g_conv')  # [bs, h, w, c']

		h = ops.conv(x, channels // 2, kernel=1, stride=1, sn=sn, scope='h_conv')  # [bs, h, w, c]
		h = ops.max_pooling(h)

		# N = h * w
		s = tf.matmul(ops.hw_flatten(g), ops.hw_flatten(f), transpose_b=True)  # # [bs, N, N]

		beta = tf.nn.softmax(s)  # attention map

		o = tf.matmul(beta, ops.hw_flatten(h))  # [bs, N, C]
		gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

		o = tf.reshape(o, shape=[batch_size, height, width, num_channels // 2])  # [bs, h, w, C]
		o = ops.conv(o, channels, kernel=1, stride=1, sn=sn, scope='attn_conv')
		x = gamma * o + x

	return x
