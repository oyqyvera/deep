#!/usr/bin/python


import matplotlib
matplotlib.use('TkAgg')
import sys,os,time
import itertools
import math,random
import glob
import tensorflow as tf 
import numpy as np 
import cv2
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
#import PIL.Image
from IPython.display import Image, display
import warnings
warnings.filterwarnings('ignore')

# Basic parameters

max_epochs = 25
base_image_path = "5_tensorflow_traffic_light_images/"
image_types = ["red", "green", "yellow"]
input_img_x = 32
input_img_y = 32
train_test_split_ratio = 0.9
batch_size = 32
checkpoint_name = "model.ckpt"

#Helper layer function

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W, stride):
	return tf.nn.conv2d(x, W, strides=[1,stride, stride, 1],padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')

#Model
x= tf.placeholder(tf.float32, shape=[None, input_img_x, input_img_y, 3])
y_ = tf.placeholder(tf.float32, shape=[None, len(image_types)])

x_image=x
#CNN layers
W_conv1 = weight_variable([3,3,3,16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 1) + b_conv1)

W_conv2 = weight_variable([3,3,16,16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 1) + b_conv2)

W_conv3 = weight_variable([3, 3, 16, 16])
b_conv3 = bias_variable([16])
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
#Pooling layer

h_pool4 = max_pool_2x2(h_conv3)
n1,n2,n3,n4 = h_pool4.get_shape().as_list()
W_fc1 = weight_variable([n2*n3*n4, 3])
b_fc1 = bias_variable([3])

#flattened pool layer to fully connected layer
h_pool4_flat = tf.reshape(h_pool4, [-1,n2*n3*n4])
y= tf.matmul(h_pool4_flat, W_fc1) + b_fc1
sess = tf.InteractiveSession()

#Loss Function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
time_start = time.time()
v_loss = least_loss = 99999999
#Load data

full_set= []
for im_type in image_types:
	for ex in glob.glob(os.path.join(base_image_path, im_type, "*")):
		im = cv2.imread(ex)
		if not im is None:
			im = cv2.resize(im, (32,32))
			one_hot_array = [0]*len(image_types)
			one_hot_array[image_types.index(im_type)] = 1
			assert(im.shape == (32,32,3))
			full_set.append((im,one_hot_array,ex))

random.shuffle(full_set)

#trani_split_test

split_index = int(math.floor(len(full_set)*train_test_split_ratio))
train_set = full_set[:split_index]	
test_set = full_set[split_index:]

#divide sets into multiple batch size
train_set_offset = len(train_set) % batch_size
test_set_offset = len(test_set) % batch_size
train_set = train_set[:len(train_set)-train_set_offset]	
test_set = test_set[:len(test_set)-test_set_offset ]
train_x,train_y,train_z = zip(*train_set)
test_x, test_y, test_z = zip(*test_set)

# interation of training in batches
print("Starting training... [{} training examples]".format(len(train_x)))
v_loss=9999999
train_loss = []
val_loss = []
for i in range(0,max_epochs):
	for tt in range(0,(len(train_x)/batch_size)):
		start_batch = batch_size*tt
		end_batch = batch_size*(tt+1)
		train_step.run(feed_dict={x: train_x[start_batch:end_batch], y_:train_y[start_batch:end_batch]})
		ex_seen = "Current epoch, examples seen: {:20} / {} \r".format(tt*batch_size,len(train_x))
		sys.stdout.write(ex_seen.format(tt*batch_size))
		sys.stdout.flush()
   
	ex_seen = "Current epoch, examples seen: {:20} / {} \r".format((tt + 1) * batch_size, len(train_x))
 	sys.stdout.write(ex_seen.format(tt * batch_size))
	sys.stdout.flush()
    
	t_loss = loss.eval(feed_dict={x:train_x, y_:train_y})
	v_loss = loss.eval(feed_dict={x:test_x, y_:test_y})

	train_loss.append(t_loss)
	val_loss.append(v_loss)

	sys.stdout.write("Epoch {:5}: loss: {:15.10f}, val. loss: {:15.10f}".format(i+1, t_loss, v_loss))
	if v_loss < least_loss:
		sys.stdout.write(", saving new beat model to {}".format(checkpoint_name))
    	least_loss=v_loss
    	filename = saver.save(sess, checkpoint_name)
	sys.stdout.write("\n")

plt.figure()
plt.xticks(np.arange(0,len(train_loss),1.0))
plt.ylabel("Loss")
plt.xlabel("Epochs")
train_line = plt.plot(range(0, len(train_loss)), train_loss, 'r', label="Train Loss")
val_line = plt.plot(range(0,len(val_loss)), val_loss, 'g', label="Validation loss") 
plt.legend()
#plt.show()

#Print out wrong examples from test set

zipped_x_y = zip(test_x,test_y)
conf_true = []
conf_pred = []
for tt in range(0, len(zipped_x_y)):
	q=zipped_x_y[tt]
	sfmax = list(sess.run(tf.nn.softmax(y.eval(feed_dict={x: [q[0]]})))[0])
	sf_ind = sfmax.index(max(sfmax))

	predicted_label = image_types[sf_ind]
	actual_label = image_types[q[1].index(max(q[1]))]

	conf_true.append(actual_label)
	conf_pred.append(predicted_label)

	if predicted_label != actual_label:
		print(" Actual: {}, predicted: {}".format(actual_label,predicted_label))
		img_path = test_z[tt]
		ex_img = Image(filename=img_path)
		display(ex_img)

#sklearn plot confusion matrix
def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	cm2 = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
	cm2 = np.around(cm2,2)
	thresh = cm.max()/2
	for i,j in itertools.product(range(cm.shape[0], cm.shape[1])):
		plt.text(j,i,str(cm[i,j])+"/"+str(cm2[i,j]),horizontalalignment="center",color="white" if cm[i,j]>thresh else "black")
	plt.tight_layout()
	plt.ylabel('True lable')
	plt.xlabel('Predicted lable')

cnf_matrix = confusion_matrix(conf_true,conf_pred)
plt.figure()
plot_confusion_matrix(cnf_matrix,classes=image_types,normalize=False,title='Normalized confusion matrix')
plt.show()




