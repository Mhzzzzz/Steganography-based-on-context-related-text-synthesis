class Config(object):
	# model
	MODEL = ["bert-brnn-rnn", "cnn-brnn-rnn", "brnn-rnn", "rnn"]
	model = MODEL[3]

	# seq2seq
	embedding_dim = 300
	num_layers = 2
	hidden_dim = 200

	# cnn encoder
	filter_sizes = [3, 4, 5]
	num_filters = 32

	# data
	dataset = "rocstory"
	max_sentence_length = 0
	max_context_length = 0
	max_context_oneline_length = 0
	ratio = 0.9
	vocab_size = 0

	# training
	epochs = 10
	batch_size = 128
	learning_rate = 0.005
	print_every_batch = 100
	test_every_batch = 1000
	GPU = "0"

	# generation
	gen_num = 2
	show = True
	save = True
