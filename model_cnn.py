import tensorflow as tf


class CNN(object):

    def build_placeholders(self, config, batch_size):
        """
        [B*(max_story_length-1),max_sentence_length+1]
        """
        self.context = tf.placeholder(shape=[batch_size, config.max_context_length, config.max_sentence_length], dtype=tf.int32, name="context")

    def __init__(self, config, batch_size, output_size):
        self.build_placeholders(config, batch_size)

        # Embedding-------------------------------------------------------------------------------------------------
        with tf.device("/cpu:0"), tf.variable_scope("embedding"):
            self.embedding = tf.Variable(
                tf.random_uniform([config.vocab_size, config.embedding_dim]),
                dtype=tf.float32, name='embedding')
            context_reshape = tf.reshape(self.context, [batch_size*config.max_context_length, config.max_sentence_length])
            context_embed = tf.nn.embedding_lookup(self.embedding, context_reshape)

        with tf.variable_scope("cnn_encoder"):
            mask_context_0 = tf.sign(tf.cast(context_reshape, tf.float32))
            mask_context_1 = tf.expand_dims(mask_context_0, -1)
            mask_context = tf.tile(mask_context_1, [1, 1, config.embedding_dim])
            context_embed_mask_0 = context_embed * mask_context
            context_embed_mask = tf.expand_dims(context_embed_mask_0, -1)

            pooled_outputs = []
            for i, filter_size in enumerate(config.filter_sizes):
                with tf.variable_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, config.embedding_dim, 1, config.num_filters]
                    W = tf.Variable(tf.random_uniform(filter_shape), dtype=tf.float32, name='cnn_W')
                    b = tf.Variable(tf.constant(0.1, shape=[config.num_filters]), dtype=tf.float32, name='cnn_b')
                    conv = tf.nn.conv2d(context_embed_mask, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    ksize = [1, config.max_sentence_length - filter_size + 1, 1, 1]
                    pooled = tf.nn.max_pool(h, ksize=ksize, strides=[1, 1, 1, 1], padding='VALID', name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = config.num_filters * len(config.filter_sizes)
            h_pool = tf.concat(pooled_outputs, 3)      # [batch_size*(config.max_story_length-1),1,1, num_filters_total]
            h_pool_flat = tf.reshape(h_pool, [batch_size*config.max_context_length, num_filters_total])

        W = tf.Variable(tf.random_uniform([num_filters_total, output_size]), dtype=tf.float32, name='bert_out_W')
        b = tf.Variable(tf.zeros([output_size]), dtype=tf.float32, name='bert_out_b')

        sentence_embedding = tf.add(tf.matmul(h_pool_flat, W), b)
        self.sentence_embedding = tf.reshape(sentence_embedding, [batch_size, config.max_context_length, output_size])


if __name__ == "__main__":
    import configs
    import data
    config = configs.Config()
    structured_train, structured_test, w2i, i2w = data.process(config, context_raw=False)
    config.vocab_size = len(i2w)
    generator = data.get_generator([structured_train], batch_size=config.batch_size, shuffle=True)
    batch = generator.__next__()
    batch = batch[0]
    context, sentence, context_length, sentence_length = data.build_batch(batch)
    cnn = CNN(config, config.batch_size, config.hidden_dim)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        feed = {
            cnn.context: context,
        }
        sentence_embedding = sess.run([cnn.sentence_embedding], feed_dict=feed)
    pass
