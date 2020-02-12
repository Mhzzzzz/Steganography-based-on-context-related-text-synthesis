import tensorflow as tf


class Seq2seq(object):

    def build_placeholders(self, batch_size, encoder_max_length, decoder_max_length):
        # 只有brnn-rnn模型用到inputs
        self.inputs = tf.placeholder(shape=[batch_size, encoder_max_length], dtype=tf.int32, name='inputs')
        self.inputs_length = tf.placeholder(shape=[batch_size], dtype=tf.int32, name='inputs_length')
        self.targets = tf.placeholder(shape=[batch_size, decoder_max_length], dtype=tf.int32, name='targets')
        self.targets_length = tf.placeholder(shape=[batch_size], dtype=tf.int32, name='targets_length')
    
    def attn(self, hidden, encoder_outputs):
        """
        :param hidden: [B,hidden_dim]
        :param encoder_outputs: [B,S,hidden_dim]
        """
        if isinstance(hidden, tuple):
            hidden = hidden[-1]
        attn_weights = tf.matmul(encoder_outputs, tf.expand_dims(hidden, 2))    # [B,S,1] 向量相乘
        attn_weights = tf.nn.softmax(attn_weights, axis=1)                        # 这个是权重系数
        context = tf.squeeze(tf.matmul(tf.transpose(encoder_outputs, [0, 2, 1]), attn_weights))     # [B,hidden_dim]
        if len(context.shape) == 1:
            context = tf.expand_dims(context, 0)
        return context  # 这个是求完加权和的
                
    def __init__(self, 
                 config,
                 batch_size,
                 decoder_max_length,
                 w2i,
                 need_encoder=True,
                 encoder_max_length=None,
                 need_encoder_embedding=False,
                 decoder_embedding=None,
                 inputs_embedded=None,
                 training_mode=True,
                 use_attention=True,):
        # situations
        if need_encoder and not need_encoder_embedding:
            assert inputs_embedded is not None
        if not training_mode:
            assert batch_size == 1
        self.build_placeholders(batch_size, encoder_max_length, decoder_max_length)

        with tf.variable_scope("encoder"):
            if need_encoder:
                # embedding----------------------------------
                if need_encoder_embedding:
                    with tf.device("/cpu:0"), tf.variable_scope("embedding"):
                        encoder_embedding = tf.Variable(
                            tf.random_uniform([config.vocab_size, config.embedding_dim]),
                            dtype=tf.float32, name='encoder_embedding')
                        inputs_embedded = tf.nn.embedding_lookup(encoder_embedding, self.inputs)

                # B-GRU--------------------------------------
                ((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state)) = \
                    tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=tf.nn.rnn_cell.GRUCell(config.hidden_dim),
                        cell_bw=tf.nn.rnn_cell.GRUCell(config.hidden_dim),
                        inputs=inputs_embedded,
                        sequence_length=self.inputs_length,     # 有多少句话
                        dtype=tf.float32,
                        time_major=False
                )
                encoder_state = tf.add(encoder_fw_final_state, encoder_bw_final_state)  # 相加 [B,hidden_dim]
                encoder_outputs = tf.add(encoder_fw_outputs, encoder_bw_outputs)        # [B,S,hidden_dim]

        with tf.variable_scope("decoder"):
            # embedding----------------------------------
            with tf.device("/cpu:0"), tf.variable_scope("embedding"):
                if not need_encoder or (not need_encoder_embedding and decoder_embedding is None):
                    decoder_embedding = tf.Variable(
                        tf.random_uniform([config.vocab_size, config.embedding_dim]),
                        dtype=tf.float32, name='decoder_embedding')
                elif not need_encoder_embedding and decoder_embedding is not None:
                    decoder_embedding = decoder_embedding
                elif need_encoder_embedding:
                    decoder_embedding = encoder_embedding
                tokens_go = tf.ones([batch_size], dtype=tf.int32, name='tokens_GO') * w2i["_GO"]
                tokens_eos = tf.ones([batch_size], dtype=tf.int32, name='tokens_EOS') * w2i["_EOS"]

            # GRU cell-----------------------------------
            with tf.variable_scope("gru_cell"):
                if config.num_layers == 1:
                    decoder_cell = tf.nn.rnn_cell.GRUCell(config.hidden_dim)
                    if need_encoder:
                        decoder_initial_state = encoder_state
                    else:
                        decoder_initial_state = decoder_cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
                else:
                    gru_cells = [tf.nn.rnn_cell.GRUCell(config.hidden_dim) for layer in range(config.num_layers)]
                    decoder_cell = tf.nn.rnn_cell.MultiRNNCell(gru_cells)
                    zero_initial_state = decoder_cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
                    if need_encoder:
                        decoder_initial_state = tuple([encoder_state]) + zero_initial_state[1:]
                    else:
                        decoder_initial_state = zero_initial_state

            # MLP weights--------------------------------
            W = tf.Variable(
                tf.random_uniform([config.hidden_dim, config.vocab_size]),
                dtype=tf.float32, name='decoder_out_W')
            b = tf.Variable(
                tf.zeros([config.vocab_size]),
                dtype=tf.float32, name="decoder_out_b")

            # GRU + MLP----------------------------------
            def loop_fn(time, previous_output, previous_state, previous_loop_state):
                if previous_state is None:                                      # time step == 0
                    initial_elements_finished = (0 >= self.targets_length)  # all False at the initial step [B]
                    initial_state = decoder_initial_state                       # last time steps cell state
                    initial_input = tf.nn.embedding_lookup(decoder_embedding, tokens_go)    # [B,embedding_dim]
                    if use_attention:
                        initial_input = tf.concat([initial_input, self.attn(initial_state, encoder_outputs)], 1)
                        # [B,embedding_dim+hidden_dim]
                    initial_output = None
                    initial_loop_state = None
                    return initial_elements_finished, initial_input, initial_state, initial_output, initial_loop_state
                else:
                    elements_finished = (time >= self.targets_length)
                    finished = tf.reduce_all(elements_finished)         # Computes the "逻辑和" 是否最终完成

                    prediction = tf.cond(finished, lambda: tokens_eos, lambda: self.targets[:, time-1])
                    input = tf.nn.embedding_lookup(decoder_embedding, prediction)
                    # tf.cond类似if else，如果完成，input就是eos，如果未完成，则是get_next_input函数的返回值
                    if use_attention:
                        input = tf.concat([input, self.attn(previous_state, encoder_outputs)], 1)
                    state = previous_state
                    output = previous_output
                    loop_state = None

                    return elements_finished, input, state, output, loop_state

            decoder_outputs_ta, decoder_state, loop_state = tf.nn.raw_rnn(decoder_cell, loop_fn)
            decoder_outputs = decoder_outputs_ta.stack()                          # [S,B,hidden_dim]
            decoder_outputs = tf.transpose(decoder_outputs, perm=[1, 0, 2])       # [S,B,hidden_dim] -> [B,S,hidden_dim]
        
            decoder_batch_size, decoder_max_steps, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
            decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, config.hidden_dim))
            decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
            decoder_logits = tf.reshape(decoder_logits_flat, (decoder_batch_size, decoder_max_steps, config.vocab_size))
            # [B,S,vocab]

        # training and prediction------------------------
        self.logits = tf.nn.softmax(decoder_logits[:, -1, :])
        self.sample = tf.multinomial(self.logits, 1)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.targets[:, :decoder_max_steps],        # [B,S]
            logits=decoder_logits,      # [B,S,vocab]
        )
        sequence_mask = tf.sequence_mask(self.targets_length, dtype=tf.float32)
        loss = loss * sequence_mask
        self.loss = tf.reduce_mean(loss)
        
        self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
