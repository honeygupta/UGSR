"""
Project: Pyramidal Edge-Maps and Attention Based Guided Thermal Super-Resolution
Author: Honey Gupta (hn.gpt1@gmail.com)

"""
import faulthandler

faulthandler.enable()
from datetime import datetime
import json
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import click
import tensorflow as tf

from skimage.io import imsave
import datasets
import data_loader
import model
import losses_FA as losses
import tensorflow.contrib.slim as slim

slim = tf.contrib.slim
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim


class Network:

	def __init__(self, output_root_dir, to_restore,
	             base_lr, max_step, dataset_name, checkpoint_dir, do_flipping):
		current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

		self._output_dir = os.path.join(output_root_dir, current_time)
		self._images_dir = os.path.join(self._output_dir, 'imgs')
		self._to_restore = to_restore
		self._base_lr = base_lr
		self._max_step = max_step
		self._dataset_name = dataset_name
		self._checkpoint_dir = checkpoint_dir
		self._do_flipping = do_flipping
		self._num_imgs_to_save = 1

	def model_setup(self):

		self.input = tf.placeholder(
			tf.float32, [
				model.BATCH_SIZE,
				model.L_IMG_HEIGHT,
				model.L_IMG_WIDTH,
				1
			], name="input")
		self.gt = tf.placeholder(
			tf.float32, [
				model.BATCH_SIZE,
				model.H_IMG_HEIGHT,
				model.H_IMG_WIDTH,
				1
			], name="gt")
		self.guide = tf.placeholder(
			tf.float32, [
				model.BATCH_SIZE,
				model.G_IMG_HEIGHT,
				model.G_IMG_WIDTH,
				model.IMG_CHANNELS
			], name="guide")

		self.global_step = slim.get_or_create_global_step()
		self.learning_rate = tf.placeholder(tf.float32, shape=[], name="lr")

		inputs = {

			'image_in': self.input,
			'image_gt': self.gt,
			'image_gd': self.guide

		}
		outputs = model.get_outputs(inputs)
		self.output = outputs['output']
		self.ualign_features = outputs['features']

	def compute_losses(self):
		rec_loss = 1 * losses.rec_loss(self.output, self.gt)
		grad_loss = losses.gradient_loss(self.output, self.gt)

		feature_alignment_loss = losses.fa_corrLoss(self.ualign_features)
		g_loss = 50 * (rec_loss + feature_alignment_loss) + 10 * grad_loss

		optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9)

		self.model_vars = tf.trainable_variables()

		g_vars = [var for var in self.model_vars if 'g_' in var.name]
		self.g_trainer = optimizer.minimize(g_loss, var_list=g_vars)

		for var in self.model_vars:
			print(var.name)

		# Summary variables for tensorboard
		self.g_loss_summ = tf.summary.scalar("g_loss", g_loss)
		self.feature_alignment_loss_summ = tf.summary.scalar("feature_alignment_loss", feature_alignment_loss)

	def save_images(self, sess, epoch):

		if not os.path.exists(self._images_dir):
			os.makedirs(self._images_dir)

		names = ['input_', 'gt_', 'guide_', 'output_']

		with open(os.path.join(
				self._output_dir, 'epoch_' + str(epoch) + '.html'
		), 'w') as v_html:
			psnr1 = 0
			ssim1 = 0

			for i in range(0, self._num_imgs_to_save):
				print("Saving image {}/{}".format(i, self._num_imgs_to_save))

				inputs = sess.run(self.inputs)

				features, output = sess.run([self.ualign_features, self.output], feed_dict={
					self.input: inputs['image_in'],
					self.gt: inputs['image_gt'],
					self.guide: inputs['image_gd']
				})

				psnr1 += psnr((np.squeeze(inputs['image_gt'][0]) + 1.0) / 2.0, (np.squeeze(output[0]) + 1.0) / 2.0)
				ssim1 += ssim((np.squeeze(inputs['image_gt'][0]) + 1.0) / 2.0, (np.squeeze(output[0]) + 1.0) / 2.0)
				tensors = [inputs['image_in'], inputs['image_gt'], inputs['image_gd'], output]

				for batch in range(model.BATCH_SIZE):
					for name, tensor in zip(names, tensors):
						image_name = os.path.split(inputs['filename'][batch].decode())[-1] + '_' + name + str(
							i) + 'batchid_' + str(batch) + ".png"
						imsave(os.path.join(self._images_dir, image_name),
						       ((np.squeeze(tensor[batch]) + 1) * 127.5).astype(np.uint8)
						       )
						v_html.write(
							"<img src=\"" +
							os.path.join('imgs', image_name) + "\">"
						)
					v_html.write("<br>")

		with open(os.path.join(self._output_dir, 'epoch_' + str(epoch) + '.txt'), 'w') as txt_file:
			txt_file.write("PSNR  " + str(psnr1 / self._num_imgs_to_save))
			txt_file.write("  SSIM  " + str(ssim1 / self._num_imgs_to_save))

		print('PSNR  ', psnr1 / self._num_imgs_to_save)
		print('SSIM  ', ssim1 / self._num_imgs_to_save)

	def train(self):
		"""Training Function."""

		self.inputs = data_loader.load_data(self._dataset_name)
		self.inputs['image_in'] = tf.image.rgb_to_grayscale(self.inputs['image_in'])
		self.inputs['image_gt'] = tf.image.rgb_to_grayscale(self.inputs['image_gt'])

		# Build the network
		self.model_setup()

		# Loss function calculations
		self.compute_losses()

		# Initializing the global variables
		init = (tf.global_variables_initializer(), tf.local_variables_initializer())
		saver = tf.train.Saver(max_to_keep=10)

		max_images = datasets.DATASET_TO_SIZES[self._dataset_name] // model.BATCH_SIZE

		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth = True

		with tf.Session(config=config) as sess:
			sess.run(init)

			# Restore the model to run the model from last checkpoint
			if self._to_restore:
				chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
				print('restoring saved checkpoint : ' + str(chkpt_fname))
				saver.restore(sess, chkpt_fname)

			writer = tf.summary.FileWriter(self._output_dir)
			if not os.path.exists(self._output_dir):
				os.makedirs(self._output_dir)

			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)

			# Training Loop
			for epoch in range(sess.run(self.global_step), self._max_step):
				print("In the epoch ", epoch)
				if epoch % 5 == 0:
					saver.save(sess, os.path.join(
						self._output_dir, "ugsr"), global_step=epoch)

				# Dealing with the learning rate as per the epoch number
				if epoch <= 100:
					curr_lr = self._base_lr
				else:
					curr_lr = self._base_lr * 0.1

				if epoch % 1 == 0:
					self.save_images(sess, epoch)

				for i in range(0, max_images):
					print("Processing batch {}/{}".format(i, max_images))

					inputs = sess.run(self.inputs)

					# Optimizing the G_A network
					_, summary_str1, summary_str2 = sess.run(
						[self.g_trainer,
						 self.g_loss_summ, self.feature_alignment_loss_summ],
						feed_dict={
							self.input:
								inputs['image_in'],
							self.gt:
								inputs['image_gt'],
							self.guide:
								inputs['image_gd'],
							self.learning_rate: curr_lr
						}
					)

					writer.add_summary(summary_str1, epoch * max_images + i)
					writer.add_summary(summary_str2, epoch * max_images + i)

					writer.flush()

				sess.run(tf.assign(self.global_step, epoch + 1))

			coord.request_stop()
			coord.join(threads)
			writer.add_graph(sess.graph)

	def test(self):
		"""Test Function."""
		print("Testing the results")

		self.inputs = data_loader.load_data(self._dataset_name)
		self.inputs['image_in'] = tf.image.rgb_to_grayscale(self.inputs['image_in'])
		self.inputs['image_gt'] = tf.image.rgb_to_grayscale(self.inputs['image_gt'])

		self.model_setup()
		saver = tf.train.Saver()
		init = tf.global_variables_initializer()
		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth = True

		with tf.Session(config=config) as sess:
			sess.run(init)

			chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
			print(chkpt_fname)
			saver.restore(sess, chkpt_fname)

			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)

			self._num_imgs_to_save = datasets.DATASET_TO_SIZES[
				self._dataset_name]
			self.save_images(sess, 0)

			coord.request_stop()
			coord.join(threads)


@click.command()
@click.option('--to_train',
              type=click.INT,
              default=0,
              help='Whether it is train or false.')
@click.option('--log_dir',
              type=click.STRING,
              default='output/',
              help='Where the data is logged to.')
@click.option('--config_filename',
              type=click.STRING,
              default='configs/test.json',
              help='The name of the configuration file.')
@click.option('--checkpoint_dir',
              type=click.STRING,
              default='checkpoint/',
              help='The name of the train/test split.')
def main(to_train, log_dir, config_filename, checkpoint_dir):
	if not os.path.isdir(log_dir):
		os.makedirs(log_dir)

	with open(config_filename) as config_file:
		config = json.load(config_file)

	to_restore = (to_train == 2)
	base_lr = float(config['base_lr']) if 'base_lr' in config else 0.0002
	max_step = int(config['max_step']) if 'max_step' in config else 200
	dataset_name = str(config['dataset_name'])
	do_flipping = bool(config['do_flipping'])

	newmodel = Network(log_dir,
	                   to_restore, base_lr, max_step,
	                   dataset_name, checkpoint_dir, do_flipping)

	if to_train > 0:
		newmodel.train()
	else:
		newmodel.test()


if __name__ == '__main__':
	main()
