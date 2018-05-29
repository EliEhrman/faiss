from __future__ import print_function
import csv
import random
import numpy as np


# import tensorflow as tf

# FLAGS = tf.flags.FLAGS

num_words = 1000 # 400000
c_rsize = 137
c_small_rsize = 99
# c_b_learn_hd = True

glove_fn = '../../data/glove/glove.6B.50d.txt'
# tf.flags.DEFINE_float('nn_lrn_rate', 0.001,
# 					  'base learning rate for nn ')
c_key_dim = 50
c_bitvec_size = 64
c_train_fraction = 0.95
c_num_centroids = 7 # should be 200
c_num_ham_winners = 10

def find_cd_closest(train_arr, test_arr):
	l_i_test_closest = []
	for test_vec in test_arr:
		cd = np.dot(train_arr, test_vec)
		l_i_test_closest.append(np.argmax(cd))
	return l_i_test_closest
		# cd_winners = np.delete(cd_winners, np.where(cd_winners == tester))

def create_baseline(nd_full_db):
	# arr_median = np.tile(np.median(nd_full_db, axis=1), (nd_full_db.shape[1], 1)).transpose()
	# return np.greater(nd_full_db, arr_median).astype(np.float32)
	arr_median = np.median(nd_full_db, axis=0)
	return np.where(nd_full_db > arr_median, np.ones_like(nd_full_db), np.zeros_like(nd_full_db)).astype(np.int)


def test(train_bin_db, test_bin_arr, l_i_test_best, rat):
	num_hits, num_poss = 0.0, 0.0
	for itest, test_vec in enumerate(test_bin_arr):
		hd = np.sum(np.where(np.not_equal(test_vec, train_bin_db), np.ones_like(train_bin_db), np.zeros_like(train_bin_db)), axis=1)
		hd_winners = np.argpartition(hd, (rat + 1))[:(rat + 1)]
		num_hits += 1.0 if np.any(hd_winners == l_i_test_best[itest]) else 0.0

	return num_hits / float(test_bin_arr.shape[0])



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

c_num_percentile_stops = 10
c_val_thresh_step_size = 100 # the val thresh is the threshold for individual input values not the aggrtegate of these threshes

def create_sel_mat(word_arr):
	# nd_percentile_stops = np.zeros((c_num_percentile_stops, c_key_dim))
	nd_min, nd_max = np.min(word_arr, axis=0), np.max(word_arr, axis=0)
	l_decile_stops = [float(decile) * (100.0 / float(c_num_percentile_stops)) for decile in range(c_num_percentile_stops+1)]
	nd_percentile = np.percentile(word_arr, l_decile_stops, axis=0)
	nd_steps = nd_percentile[-1] - nd_percentile[0] / c_val_thresh_step_size
	nd_val_thresh = np.random.choice(a=c_num_percentile_stops+1, size=(c_key_dim, c_bitvec_size))
	# nd_thresh = np.zeros((c_key_dim, c_bitvec_size))
	# for ival in range(c_key_dim):
	# 	nd_thresh[ival, :] = np.take(nd_percentile[:, ival], nd_val_thresh[ival, :])
	nd_thresh_ivals = np.asarray([np.take(nd_percentile[:, ival], nd_val_thresh[ival, :]) for ival in range(c_key_dim)])
	nd_thresh_sum_thresh = np.sum(1.0 - (nd_val_thresh.astype(float) / float(c_num_percentile_stops)), axis=0)
	# return (np.random.rand(c_key_dim, c_bitvec_size), np.full(c_key_dim, c_bitvec_size/2))
	return (nd_thresh_ivals, nd_thresh_sum_thresh, nd_steps)

def create_bit_db(word_arr, sel_mat):
	numrecs = word_arr.shape[0]
	bitcomps, thresh, nd_steps = sel_mat
	a = np.tile(np.expand_dims(word_arr, axis=-1), reps=[1, c_bitvec_size])
	b = np.tile(np.expand_dims(bitcomps, axis=0), reps=[numrecs, 1, 1])
	c = np.where(a>b, np.ones_like(a), np.zeros_like(a))
	# g = np.sum(c, axis=1)
	d = np.where(np.sum(c, axis=1) > thresh, np.ones((numrecs, c_bitvec_size), dtype=np.uint8),
				 np.zeros((numrecs, c_bitvec_size), dtype=np.uint8))
	# e = np.sum(d, axis=0).astype(float) / float(numrecs)
	# for iiter in xrange(300):
	# 	thresh += (e - 0.5) * 0.01
	# 	d = np.where(np.sum(c, axis=1) > thresh, np.ones((numrecs, c_bitvec_size), dtype=np.uint8),
	# 				 np.zeros((numrecs, c_bitvec_size), dtype=np.uint8))
	# 	e = np.sum(d, axis=0).astype(float) / float(numrecs)
	# 	f = np.sum(d, axis=0)
	# 	g = np.sum(c, axis=1)
	return d


def main():
	word_dict, word_arr = load_word_dict()
	num_recs_total = word_arr.shape[0]
	train_limit = int(num_recs_total * c_train_fraction)
	nd_train_recs = word_arr[:train_limit, :]
	nd_q_recs = word_arr[train_limit:, :]
	l_i_test_closest = find_cd_closest(nd_train_recs, nd_q_recs)
	nd_bin_db, nd_bin_q = create_baseline(nd_train_recs), create_baseline(nd_q_recs)
	rat10 = test(nd_bin_db, nd_bin_q, l_i_test_closest, 10)
	# rat100 = test(nd_bin_db, nd_bin_q, l_i_test_closest, 100)
	# rat1000 = test(nd_bin_db, nd_bin_q, l_i_test_closest, 1000)
	# print('r@: 10, 100, 1000:', rat10, rat100, rat1000)
	print('baseline r@: 10:', rat10)
	sel_mat = create_sel_mat(nd_train_recs)
	nd_bin_db, nd_bin_q = create_bit_db(nd_train_recs, sel_mat), create_bit_db(nd_q_recs, sel_mat)
	rat10 = test(nd_bin_db, nd_bin_q, l_i_test_closest, 10)
	print('starting r@: 10:', rat10)
	return


main()
print('done')


