import tensorflow as tf


def _phase_shift(I, r):
	bsize, a, b, c = I.get_shape().as_list()
	bsize = tf.shape(I)[0]  # Handling Dimension(None) type for undefined batch dim
	X = tf.reshape(I, (bsize, a, b, r, r))
	X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
	X = tf.split(1, a, X)  # a, [bsize, b, r, r]
	X = tf.concat(2, [tf.squeeze(x, axis=1) for x in X])  # bsize, b, a*r, r
	X = tf.split(1, b, X)  # b, [bsize, a*r, r]
	X = tf.concat(2, [tf.squeeze(x, axis=1) for x in X])  # bsize, a*r, b*r
	return tf.reshape(X, (bsize, a * r, b * r, 1))


def PS(X, r, color=False):
	if color:
		Xc = tf.split(3, 3, X)
		X = tf.concat(3, [_phase_shift(x, r) for x in Xc])
	else:
		X = _phase_shift(X, r)
	return X


def lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):
	with tf.variable_scope(name):
		if alt_relu_impl:
			f1 = 0.5 * (1 + leak)
			f2 = 0.5 * (1 - leak)
			return f1 * x + f2 * abs(x)
		else:
			return tf.maximum(x, leak * x)


def instance_norm(x):
	with tf.variable_scope("instance_norm"):
		epsilon = 1e-5
		mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
		scale = tf.get_variable('scale', [x.get_shape()[-1]],
		                        initializer=tf.truncated_normal_initializer(
			                        mean=1.0, stddev=0.02
		                        ))
		offset = tf.get_variable(
			'offset', [x.get_shape()[-1]],
			initializer=tf.constant_initializer(0.0)
		)
		out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset

		return out


def general_conv2d(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02,
                   padding="VALID", name="conv2d", do_norm=True, do_relu=True,
                   relufactor=0):
	with tf.variable_scope(name):

		conv = tf.contrib.layers.conv2d(
			inputconv, o_d, f_w, s_w, padding,
			activation_fn=None,
			weights_initializer=tf.truncated_normal_initializer(
				stddev=stddev
			),
			biases_initializer=tf.constant_initializer(0.0)
		)
		if do_norm:
			conv = instance_norm(conv)

		if do_relu:
			if (relufactor == 0):
				conv = tf.nn.relu(conv, "relu")
			else:
				conv = lrelu(conv, relufactor, "lrelu")

		return conv


def general_deconv2d(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1,
                     stddev=0.02, padding="VALID", name="deconv2d",
                     do_norm=True, do_relu=True, relufactor=0):
	with tf.variable_scope(name):

		conv = tf.contrib.layers.conv2d_transpose(
			inputconv, o_d, [f_h, f_w],
			[s_h, s_w], padding,
			activation_fn=None,
			weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
			biases_initializer=tf.constant_initializer(0.0)
		)

		if do_norm:
			conv = instance_norm(conv)

		if do_relu:
			if (relufactor == 0):
				conv = tf.nn.relu(conv, "relu")
			else:
				conv = lrelu(conv, relufactor, "lrelu")

		return conv
