import tensorflow as tf
import image

img = image.run()
saver = tf.train.Saver()

def run():
	with tf.Session() as sess:
		saver.restore('./models/mnist_model.ckpt')

	