"""
Project: Unaligned Guided Thermal Image Super-Resolution
Author: Honey Gupta (hn.gpt1@gmail.com)

"""

import csv
import json
import os
from datetime import datetime

import click
import numpy as np
import tensorflow as tf
from skimage.io import imsave
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import model
import datasets
import losses
import tensorflow.contrib.slim as slim

import cv2

slim = tf.contrib.slim


def load_data(filename):
	list1 = []
	list2 = []
	list3 = []
	csvfile = open(filename, 'r')
	spamreader = csv.reader(csvfile, delimiter=',')
	for row in spamreader:
		list1.append(row[0])
		list2.append(row[1])
		list3.append(row[2])
	return list1, list2, list3


class Data(object):
	def __init__(self, list1, list2, list3, bs, shuffle=False):
		self.list1 = list1
		self.list2 = list2
		self.list3 = list3

		self.bs = bs
		self.index = 0
		self.number = len(self.list1)
		self.index_total = range(self.number)
		self.shuffle = shuffle
		if self.shuffle:
			self.index_total = np.random.permutation(self.number)

	def next_batch(self):
		start = self.index
		self.index += self.bs
		if self.index > self.number:
			self.index = 0
			start = self.index
			self.index += self.bs
		end = self.index
		img1_batch = []
		img2_batch = []
		img3_batch = []

		name = []
		for i in range(start, end):
			im1 = cv2.imread(self.list1[self.index_total[i]], 0)
			im1 = (im1.astype(np.float32) / 127.5) - 1.0
			im1 = np.expand_dims(im1, 2)

			im2 = cv2.imread(self.list2[self.index_total[i]], 0)
			im2 = (im2.astype(np.float32) / 127.5) - 1.0
			im2 = np.expand_dims(im2, 2)

			# im2 = np.load(self.list2[self.index_total[i]])
			# im2 = np.array(im2).astype(np.float32)
			# im2 = im2 / model.G_IMG_WIDTH
			# im2 = np.expand_dims(im2,2)

			im3 = cv2.imread(self.list3[self.index_total[i]])
			im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)
			# l = im3[:, :, 0:1]
			im3 = (im3.astype(np.float32) / 127.5) - 1.0

			txt = self.list1[self.index_total[i]]
			name.append(os.path.basename(txt))

			img1_batch.append(im1)
			img2_batch.append(im2)
			img3_batch.append(im3)

		return np.array(img1_batch), np.array(img2_batch), np.array(img3_batch), name


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
			'image_gd': self.guide,

		}
		outputs = model.get_outputs(inputs)
		self.output = outputs['output']
		self.refined_disp = outputs['disp']
		self.ualign_features = outputs['features']
		self.warped = outputs['warped']
		self.confidence = outputs['confidence']

	def compute_losses(self):

		rec_loss = losses.rec_loss(self.output, self.gt)
		print('self.warped', self.warped)
		print('self.gt', self.gt)

		grad_loss = losses.gradient_loss(self.output, self.gt)
		g_loss = 10 * (rec_loss) + 2 * grad_loss + 0.1 * losses.l0_norm(self.refined_disp)

		optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9)

		self.model_vars = tf.trainable_variables()

		g_vars = [var for var in self.model_vars if 'g_' in var.name]
		self.g_trainer = optimizer.minimize(g_loss, var_list=g_vars)

		# Summary variables for tensorboard
		self.g_loss_summ = tf.summary.scalar("g_loss", g_loss)

	def save_images(self, sess, epoch):

		if not os.path.exists(self._images_dir):
			os.makedirs(self._images_dir)

		names = ['input_', 'gt_', 'guide_', 'output_', 'warped_', 'conf_']

		with open(os.path.join(
				self._output_dir, 'epoch_' + str(epoch) + '.html'
		), 'w') as v_html:
			psnr1 = 0
			ssim1 = 0
			count = 0
			for i in range(0, self._num_imgs_to_save):
				print("Saving image {}/{}".format(i, self._num_imgs_to_save))

				x1_t, x2_t, x3_t, name1 = self.dataset.next_batch()

				output, disparity, warped, conf, features = sess.run(
					[self.output, self.refined_disp, self.warped, self.confidence, self.refined_disp], feed_dict={
						self.input: x1_t,
						self.gt: x2_t,
						self.guide: x3_t,
					})
				tensors = [x1_t, x2_t, x3_t, output, warped, conf]
				for batch in range(model.BATCH_SIZE):
					psnr1 += psnr(np.squeeze(x2_t[batch]), np.squeeze(output[batch]), data_range=2)
					ssim1 += ssim(np.squeeze(x2_t[batch]), np.squeeze(output[batch]), data_range=2)
					count += 1
					for name, tensor in zip(names, tensors):
						image_name = os.path.split(name1[batch])[-1] + '_' + name + str(i) + 'batchid_' + str(
							batch) + ".png"
						if name == 'conf_':
							imsave(os.path.join(self._images_dir, image_name),
							       ((np.squeeze(tensor[batch]) + 1) * 127.5).astype(np.uint8)
							       )
						elif name == 'warped_':
							tensor = tensor[:, :, :, 0]
							t = ((np.squeeze(tensor[batch]) + 1) * 127.5).astype(np.uint8)
							imsave(os.path.join(self._images_dir, image_name), t
							       )
						else:
							imsave(os.path.join(self._images_dir, image_name),
							       ((np.squeeze(tensor[batch]) + 1) * 127.5).astype(np.uint8)
							       )
						v_html.write(
							"<img src=\"" +
							os.path.join('imgs', image_name) + "\">"
						)
					v_html.write("<br>")

		with open(os.path.join(self._output_dir, 'epoch_' + str(epoch) + '.txt'), 'w') as txt_file:
			txt_file.write("PSNR  " + str(psnr1 / count))
			txt_file.write("  SSIM  " + str(ssim1 / count))

		print('PSNR  ', psnr1 / count)
		print('SSIM  ', ssim1 / count)

	def train(self):
		"""Training Function."""

		list1, list2, list3 = load_data(datasets.PATH_TO_CSV[self._dataset_name])
		self.dataset = Data(list1, list2, list3, bs=model.BATCH_SIZE, shuffle=True)

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
				vars_list = tf.train.list_variables(chkpt_fname)
				vnames = []
				for var in vars_list:
					vnames.append(var[0])
				vlist = []
				for var in self.model_vars:
					if var.name[:-2] in vnames:
						print(var.name)
						vlist.append(var)

				saver1 = tf.train.Saver(vlist)
				saver1.restore(sess, chkpt_fname)

			writer = tf.summary.FileWriter(self._output_dir)
			if not os.path.exists(self._output_dir):
				os.makedirs(self._output_dir)

			# Training Loop
			for epoch in range(sess.run(self.global_step), self._max_step):
				print("In the epoch ", epoch)
				if epoch % 10 == 0:
					saver.save(sess, os.path.join(
						self._output_dir, "ugsr"), global_step=epoch)

				# Dealing with the learning rate as per the epoch number
				if epoch <= 100:
					curr_lr = self._base_lr
				else:
					curr_lr = self._base_lr * 0.1

				if epoch % 5 == 0:
					self.save_images(sess, epoch)

				for i in range(0, max_images):
					print("Processing batch {}/{}".format(i, max_images))

					x1_t, x2_t, x3_t, _ = self.dataset.next_batch()

					# Optimizing the G_A network
					_, summary_str1 = sess.run(
						[self.g_trainer,
						 self.g_loss_summ],
						feed_dict={
							self.input:
								x1_t,
							self.gt:
								x2_t,
							self.guide: x3_t,
							self.learning_rate: curr_lr
						}
					)

					writer.add_summary(summary_str1, epoch * max_images + i)

					writer.flush()

				sess.run(tf.assign(self.global_step, epoch + 1))

			writer.add_graph(sess.graph)

	def test(self):
		"""Test Function."""
		print("Testing the results")

		list1, list2, list3 = load_data(datasets.PATH_TO_CSV[self._dataset_name])
		self.dataset = Data(list1, list2, list3, bs=model.BATCH_SIZE, shuffle=False)

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

			self._num_imgs_to_save = datasets.DATASET_TO_SIZES[
				self._dataset_name]
			self.save_images(sess, 0)


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
              default='../configs/text.json',
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
