"""
Project: Unaligned Guided Thermal Image Super-Resolution
Author: Honey Gupta (hn.gpt1@gmail.com)

"""

import csv
import os
import random

import click

import datasets


def create_list(foldername, fulldir=True, suffix=".png", multifolder=False, ):
	file_list_tmp = os.listdir(foldername)
	file_list_tmp.sort()
	file_list = []
	if fulldir:
		if multifolder:
			for folder in file_list_tmp:
				temp_list = os.listdir(os.path.join(foldername, folder))
				temp_list.sort()
				for item in temp_list:
					if item.endswith(suffix):
						file_list.append(os.path.join(foldername, folder, item))
		else:
			for item in file_list_tmp:
				if item.endswith(suffix):
					file_list.append(os.path.join(foldername, item))
	else:
		for item in file_list_tmp:
			if item.endswith(suffix):
				file_list.append(item)
	return file_list


@click.command()
@click.option('--input_image',
              type=click.STRING,
              default='../datasets/flir_adas/ther_64_80',
              help='The path to the input images.')
@click.option('--ground_truth',
              type=click.STRING,
              default='../datasets/flir_adas/4x/ther',
              help='The path to the ground truth images.')
@click.option('--guide_image',
              type=click.STRING,
              default='../datasets/flir_adas/4x/rgb',
              help='The path to the guide images.')
@click.option('--dataset_name',
              type=click.STRING,
              default='flir_test',
              help='The name of the dataset.')
@click.option('--do_shuffle',
              type=click.BOOL,
              default=False,
              help='Whether to shuffle images when creating the dataset.')
def create_dataset(input_image, ground_truth, guide_image,
                   dataset_name, do_shuffle):
	list_a = create_list(input_image, True,
	                     datasets.DATASET_TO_IMAGETYPE[dataset_name], False)
	list_b = create_list(ground_truth, True,
	                     '.png', False)
	list_c = create_list(guide_image, True,
	                     '.png', False)

	output_path = datasets.PATH_TO_CSV[dataset_name]
	num_rows = datasets.DATASET_TO_SIZES[dataset_name]
	all_data_tuples = []
	for i in range(num_rows):
		all_data_tuples.append((
			list_a[i % len(list_a)],
			list_b[i % len(list_b)],
			list_c[i % len(list_c)]
		))

	if do_shuffle is True:
		random.shuffle(all_data_tuples)
	with open(output_path, 'w') as csv_file:
		csv_writer = csv.writer(csv_file)
		for data_tuple in enumerate(all_data_tuples):
			csv_writer.writerow(list(data_tuple[1]))


if __name__ == '__main__':
	create_dataset()
