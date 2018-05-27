from __future__ import print_function
import csv
import random
import numpy as np


import tensorflow as tf

FLAGS = tf.flags.FLAGS

num_words = 10000 # 10000 # 400000
c_rsize = 137
c_small_rsize = 99
# c_b_learn_hd = True

glove_fn = '../../data/glove/glove.6B.50d.txt'
tf.flags.DEFINE_float('nn_lrn_rate', 0.001,
					  'base learning rate for nn ')
c_key_dim = 50
c_train_fraction = 0.5
c_num_centroids = 7 # should be 200

def load_word_dict():
	global g_word_vec_len
	glove_fh = open(glove_fn, 'rb')
	glove_csvr = csv.reader(glove_fh, delimiter=' ', quoting=csv.QUOTE_NONE)

	word_dict = {}
	word_arr = []
	for irow, row in enumerate(glove_csvr):
		word = row[0]
		vec = [float(val) for val in row[1:]]
		vec = np.array(vec, dtype=np.float32)
		en = np.linalg.norm(vec, axis=0)
		vec = vec / en
		word_dict[word] = vec
		word_arr.append(vec)
		if irow > num_words:
			break
	# print(row)

	glove_fh.close()
	g_word_vec_len = len(word_dict['the'])
	random.shuffle(word_arr)
	return word_dict, np.array(word_arr)

def make_db_cg(numrecs):
	v_db_norm = tf.Variable(tf.zeros([numrecs, c_key_dim], dtype=tf.float32), trainable=False)
	ph_db_norm = tf.placeholder(dtype=tf.float32, shape=[numrecs, c_key_dim], name='ph_db_norm')
	op_db_norm_assign = tf.assign(v_db_norm, ph_db_norm, name='op_db_norm_assign')
	return v_db_norm, ph_db_norm, op_db_norm_assign

def make_per_batch_init_cg(numrecs, v_db_norm, num_centroids):
	# The goal is to cluster the convolution vectors so that we can perform dimension reduction
	# KMeans implementation
	# Intitialize the centroids indicies. Shape=[num_centroids]
	t_centroids_idxs_init = tf.random_uniform([num_centroids], 0, numrecs - 1, dtype=tf.int32,
											  name='t_centroids_idxs_init')
	# Get the centroids variable ready. Must persist between loops. Shape=[c_num_convs*c_num_files_per_batch, c_kernel_size]
	v_centroids = tf.Variable(tf.zeros([num_centroids, c_key_dim], dtype=tf.float32), name='v_centroids')
	# Create actual centroids as seeds. Shape=[num_centroids, c_kernel_size]
	op_centroids_init = tf.assign(v_centroids, tf.gather(v_db_norm, t_centroids_idxs_init, name='op_centroids_init'))

	return v_centroids, op_centroids_init

def prep_learn():
	# t_y_db, l_W_db, l_W_q, l_batch_assigns, t_err, op_train_step = \
	# 	dmlearn.prep_learn(ivec_dim_dict_db, ivec_dim_dict_q, ivec_arr_db, ivec_arr_q, match_pairs, mismatch_pairs)
	# sess, saver = dmlearn.init_learn(l_W_db + l_W_q)
	# do_set_eval(sess, input_db, output_db,  t_y_db, input_eval,
	# 			event_results_eval, event_result_id_arr)
	pass

def learn(nd_train_recs):
	numrecs = nd_train_recs.shape[0]
	# v_full_db = tf.Variable(tf.zeros([numrecs, c_key_dim], dtype=tf.float32), name='v_full_db')
	v_db_norm, ph_db_norm, op_db_norm_assign = make_db_cg(numrecs)
	v_centroids, op_centroids_init = make_per_batch_init_cg(numrecs, v_db_norm, c_num_centroids)

	pass

def main():
	word_dict, word_arr = load_word_dict()
	num_recs_total = word_arr.shape[0]
	train_limit = int(num_recs_total * c_train_fraction)
	nd_train_recs = word_arr[:train_limit, :]
	nd_q_recs = word_arr[train_limit:, :]
	learn(train_limit)
	return


main()
print('done')


