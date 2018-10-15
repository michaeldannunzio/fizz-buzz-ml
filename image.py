import os
import skimage.io
import numpy as np

def load_data():
	img = skimage.io.imread('./assets/digits.png')
	return img

def get_rows(img):
	rows = []
	for i in range(100):
		s = i * 28
		e = (i + 1) * 28
		rows.append(img[s:e, :])
	return rows

def get_nums(img):
	nums = []
	for row in img:
		for i in range(3):
			s = i * 28
			e = (i + 1) * 28
			nums.append(row[:, s:e])
	return nums

def vectorize_img(img):
	vecs = []
	for i in img:
		vecs.append(i.flatten().reshape(1, 784))
	return vecs

def run():
	img = load_data()
	img = get_rows(img)
	img = get_nums(img)
	img = vectorize_img(img)
	return img
