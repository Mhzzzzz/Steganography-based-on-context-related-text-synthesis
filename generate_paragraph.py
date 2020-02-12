import tensorflow as tf
import os
import random
import time
import numpy as np

import model_seq2seq
import model_cnn
import configs
import data
import Huffman_Encoding


if __name__ == "__main__":
    config = configs.Config()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU
    tf_config = tf.ConfigProto()
    tf_config.allow_soft_placement = True
    tf_config.gpu_options.allow_growth = True

    print("(1) load data ......\n")
    structured_train, _, w2i, i2w = data.process(config, context_min=-1, need_context=True, context_oneline=False)
    config.vocab_size = len(w2i)
    generator_train = data.get_generator([structured_train], batch_size=1, shuffle=True)

    print("(2) build model ......")
    assert config.model in config.MODEL
    if config.model == "cnn-brnn-rnn":
        print("## building cnn-brnn-rnn\n")
        cnn = model_cnn.CNN(config, 1, config.hidden_dim)
        seq2seq = model_seq2seq.Seq2seq(config,
                                        1,
                                        config.max_sentence_length,
                                        w2i,
                                        inputs_embedded=cnn.sentence_embedding,
                                        decoder_embedding=cnn.embedding)
    elif config.model == "brnn-rnn":
        print("## building brnn-rnn\n")
        seq2seq = model_seq2seq.Seq2seq(config,
                                        1,
                                        config.max_sentence_length,
                                        w2i,
                                        need_encoder_embedding=True,
                                        encoder_max_length=90)
    elif config.model == "rnn":
        print("## building rnn\n")
        seq2seq = model_seq2seq.Seq2seq(config,
                                        1,
                                        config.max_sentence_length,
                                        w2i,
                                        need_encoder=False,
                                        use_attention=False)

    print("(3) generate steganographic paragraph ......")
    try:
        os.makedirs("generate_paragraph/" + config.model + "/")
    except:
        pass
    with open("bit_stream/bit_stream.txt", "r", encoding="utf8") as f:
        bit_stream = f.read()
    bit_index = random.randint(0, 1000)

    with tf.Session(config=tf_config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint("checkpoint/"+config.model+"/"))

        for bit_num in range(1, 6):
            generation_list = []
            for i in range(config.gen_num):
                batch_data = generator_train.__next__()
                batch_data = batch_data[0][0]

                if config.model == "cnn-brnn-rnn":
                    context = [[batch_data["context"][0]] + [[w2i["_PAD"]]*config.max_sentence_length for _ in range(config.max_context_length-1)]]
                    context_length = [1]
                elif config.model == "brnn-rnn":
                    context = [[batch_data["context"][0]]]
                    context_oneline = [batch_data["context"][0] + [w2i["_PAD"]]*(90 - len(batch_data["context"][0]))]
                    context_oneline_length = [sum([1 for _ in context_oneline[0] if _ != w2i["_PAD"]])] # including eos
                elif config.model == "rnn":
                    context = [[batch_data["context"][0]]]

                bit_insert = ""

                for k in range(1, batch_data["context_length"]+1):            # start from the second sentence
                    sentence = [[w2i["_PAD"]]*config.max_sentence_length]
                    sentence_length = [1]

                    for j in range(0, config.max_sentence_length):
                        if config.model == "cnn-brnn-rnn":
                            feed = {
                                cnn.context: context,
                                seq2seq.inputs_length: context_length,
                                seq2seq.targets: sentence,
                                seq2seq.targets_length: sentence_length,
                            }
                        elif config.model == "brnn-rnn":
                            feed = {
                                seq2seq.inputs: context_oneline,
                                seq2seq.inputs_length: context_oneline_length,
                                seq2seq.targets: sentence,
                                seq2seq.targets_length: sentence_length,
                            }
                        elif config.model == "rnn":
                            feed = {
                                seq2seq.targets: sentence,
                                seq2seq.targets_length: sentence_length,
                            }

                        if j == -1:
                            sample = sess.run(seq2seq.sample, feed_dict=feed)[0][0]
                        else:
                            logits = sess.run(seq2seq.logits, feed_dict=feed)[0]
                            p = {}
                            for i in range(len(logits)):
                                p[i] = logits[i]
                            prob_sort = sorted(p.items(), key=lambda x: x[1], reverse=True)

                            m = 2 ** int(bit_num)
                            word_prob = []
                            for i in range(m + 1):
                                if len(word_prob) == m:
                                    break
                                if prob_sort[i][0] == w2i["_UNK"]:
                                    continue
                                else:
                                    word_prob.append(prob_sort[i])

                            nodes = Huffman_Encoding.createNodes([item[1] for item in word_prob])
                            root = Huffman_Encoding.createHuffmanTree(nodes)
                            codes = Huffman_Encoding.huffmanEncoding(nodes, root)
                            # print codes
                            for i in range(m):
                                if bit_stream[bit_index:bit_index + i + 1] in codes:
                                    code_index = codes.index(bit_stream[bit_index:bit_index + i + 1])
                                    sample = int(word_prob[code_index][0])
                                    break

                        sentence[0][j] = sample
                        sentence_length[0] = sentence_length[0] + 1
                        if sample == w2i["_EOS"]:
                            break
                        bit_insert += bit_stream[bit_index:bit_index + i + 1]
                        bit_index = bit_index + i + 1

                    if config.model == "cnn-brnn-rnn":
                        context[0][k] = sentence[0]
                        context_length[0] = context_length[0] + 1
                    elif config.model == "brnn-rnn":
                        pure_sentence = [_ for _ in sentence[0] if i2w[_] not in ["_PAD", "_EOS"]]
                        context[0].append(pure_sentence)
                        context_oneline[0][context_oneline_length[0]-1:context_oneline_length[0]+len(pure_sentence)] = \
                            pure_sentence + [w2i["_EOS"]]
                        context_oneline_length[0] = context_oneline_length[0] + len(pure_sentence)
                    elif config.model == "rnn":
                        context[0].append(sentence[0])

                batch_data["generation"] = context[0]
                batch_data["bit_insert"] = bit_insert
                generation_list.append(batch_data)

                if config.show:
                    print("cover:")
                    for s in batch_data["context"]:
                        s_str = " ".join([i2w[num] for num in s if i2w[num] not in ["_PAD", "_EOS"]])
                        if s_str != "":
                            print(s_str)
                    print(" ".join([i2w[num] for num in batch_data["sentence"] if i2w[num] not in ["_PAD", "_EOS"]]))
                    print("generation:")
                    for s in batch_data["generation"]:
                        s_str = " ".join([i2w[num] for num in s if i2w[num] not in ["_PAD", "_EOS"]])
                        print(s_str)
                    print("")

            if config.save:
                time_str = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
                np.save("generate_paragraph/" + config.model + "/" + str(bit_num) + "bit_" + time_str + ".npy", generation_list)
