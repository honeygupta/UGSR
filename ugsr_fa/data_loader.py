"""
Project: Unaligned Guided Thermal Image Super-Resolution
Author: Honey Gupta (hn.gpt1@gmail.com)

"""
import tensorflow as tf

import datasets
import model


def random_crop(img1, img2, size=[model.H_IMG_HEIGHT, model.H_IMG_WIDTH]):
	combined = tf.concat([img1, img2], axis=2)
	image_shape = tf.shape(img1)
	combined_pad = tf.image.pad_to_bounding_box(
		combined, 0, 0,
		tf.maximum(size[0], image_shape[0]),
		tf.maximum(size[1], image_shape[1]))

	img1_dim = 3
	img2_dim = 3

	combined_crop = tf.random_crop(
		combined_pad,
		size=tf.concat([size, [img1_dim + img2_dim]],
		               axis=0))
	return (combined_crop[:, :, :img1_dim], combined_crop[:, :, img1_dim:])


def _load_samples(csv_name, image_type):
	filename_queue = tf.train.string_input_producer([csv_name])

	reader = tf.TextLineReader()
	_, csv_filename = reader.read(filename_queue)

	record_defaults = [tf.constant([], dtype=tf.string),
	                   tf.constant([], dtype=tf.string),
	                   tf.constant([], dtype=tf.string)]

	filename_in, filename_gt, filename_gd = tf.decode_csv(csv_filename, record_defaults=record_defaults)

	file_contents_in = tf.read_file(filename_in)
	file_contents_gt = tf.read_file(filename_gt)
	file_contents_gd = tf.read_file(filename_gd)

	if image_type == '.jpg':
		image_decoded_A = tf.image.decode_jpeg(
			file_contents_in, channels=model.IMG_CHANNELS)
		image_decoded_B = tf.image.decode_jpeg(
			file_contents_gt, channels=model.IMG_CHANNELS)
		image_decoded_C = tf.image.decode_jpeg(
			file_contents_gd, channels=model.IMG_CHANNELS)


	elif image_type == '.png':
		image_decoded_A = tf.image.decode_png(
			file_contents_in, channels=model.IMG_CHANNELS, dtype=tf.uint8)
		image_decoded_B = tf.image.decode_png(
			file_contents_gt, channels=model.IMG_CHANNELS, dtype=tf.uint8)
		image_decoded_C = tf.image.decode_png(
			file_contents_gd, channels=model.IMG_CHANNELS, dtype=tf.uint8)

	return image_decoded_A, image_decoded_B, image_decoded_C, filename_gt


def load_data(dataset_name,
              do_shuffle=True, do_flipping=False):
	if dataset_name not in datasets.DATASET_TO_SIZES:
		raise ValueError('split name %s was not recognized.'
		                 % dataset_name)

	csv_name = datasets.PATH_TO_CSV[dataset_name]

	image_in, image_gt, image_gd, filename = _load_samples(csv_name, datasets.DATASET_TO_IMAGETYPE[dataset_name])

	image_in.set_shape([model.L_IMG_HEIGHT, model.L_IMG_WIDTH, 3])
	image_gt.set_shape([model.H_IMG_HEIGHT, model.H_IMG_WIDTH, 3])
	image_gd.set_shape([model.G_IMG_HEIGHT, model.G_IMG_WIDTH, 3])

	inputs = {
		'image_in': image_in,
		'image_gt': image_gt,
		'image_gd': image_gd,
		'filename': filename
	}

	inputs['image_in'] = tf.cast(inputs['image_in'], tf.float32)
	inputs['image_gt'] = tf.cast(inputs['image_gt'], tf.float32)
	inputs['image_gd'] = tf.cast(inputs['image_gd'], tf.float32)

	inputs['image_in'] = tf.subtract(tf.div(inputs['image_in'], 127.5), 1)
	inputs['image_gt'] = tf.subtract(tf.div(inputs['image_gt'], 127.5), 1)
	inputs['image_gd'] = tf.subtract(tf.div(inputs['image_gd'], 127.5), 1)

	inputs['image_gt'], inputs['image_gd'], inputs['image_in'], inputs['filename'] = tf.train.batch([inputs['image_gt'],
	                                                                                                 inputs['image_gd'],
	                                                                                                 inputs['image_in'],
	                                                                                                 inputs[
		                                                                                                 'filename']],
	                                                                                                model.BATCH_SIZE)

	return inputs
