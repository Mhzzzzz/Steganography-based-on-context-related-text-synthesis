import tensorflow as tf
import time
import os

import model_seq2seq
import model_cnn
import configs
import data

    
if __name__ == "__main__":
    config = configs.Config()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU
    tf_config = tf.ConfigProto()
    tf_config.allow_soft_placement = True
    tf_config.gpu_options.allow_growth = True

    print("(1) load data ......\n")
    if config.model == "cnn-brnn-rnn":
        structured_train, structured_test, w2i, i2w = data.process(config, need_context=True, context_oneline=False)
    elif config.model == "brnn-rnn":
        structured_train, structured_test, w2i, i2w = data.process(config, need_context=True, context_oneline=True)
    elif config.model == "rnn":
        structured_train, structured_test, w2i, i2w = data.process(config, need_context=False, context_oneline=False)

    print("## training set size: " + str(len(structured_train)))
    print("## test set size: " + str(len(structured_test)))
    print("## processed vocab size: " + str(len(w2i)) + "\n")
    config.vocab_size = len(w2i)
    generator_train = data.get_generator([structured_train], batch_size=config.batch_size, shuffle=True)
    
    print("(2) build model ......")
    assert config.model in config.MODEL
    if config.model == "cnn-brnn-rnn":
        print("## building cnn-brnn-rnn\n")
        cnn = model_cnn.CNN(config, config.batch_size, config.hidden_dim)
        seq2seq = model_seq2seq.Seq2seq(config,
                                        config.batch_size,
                                        config.max_sentence_length,
                                        w2i,
                                        inputs_embedded=cnn.sentence_embedding,
                                        decoder_embedding=cnn.embedding)
    elif config.model == "brnn-rnn":
        print("## building brnn-rnn\n")
        seq2seq = model_seq2seq.Seq2seq(config,
                                        config.batch_size,
                                        config.max_sentence_length,
                                        w2i,
                                        need_encoder_embedding=True,
                                        encoder_max_length=config.max_context_oneline_length)
    elif config.model == "rnn":
        print("## building rnn\n")
        seq2seq = model_seq2seq.Seq2seq(config,
                                        config.batch_size,
                                        config.max_sentence_length,
                                        w2i,
                                        need_encoder=False,
                                        use_attention=False)

    print("(3) run model ......")
    try:
        os.makedirs("checkpoint/"+config.model+"/")
        print("## making checkpoint directory")
    except:
        pass
    with tf.Session(config=tf_config) as sess:
        saver = tf.train.Saver(max_to_keep=5)
        sess.run(tf.global_variables_initializer())

        batches = (config.epochs * len(structured_train)) // config.batch_size
        batches_test = len(structured_test) // config.batch_size
        print("## train batches: " + str(batches))
        print("## test batches: " + str(batches_test))

        losses = []
        test_losses = []
        total_loss = 0
        for batch in range(batches):
            batch_data = generator_train.__next__()
            batch_data = batch_data[0]
            if config.model == "cnn-brnn-rnn":
                context, _, sentence, context_length, _, sentence_length = \
                    data.build_batch(batch_data, context=True, context_oneline=False)
            elif config.model == "brnn-rnn":
                _, context_oneline, sentence, _, context_oneline_length, sentence_length = \
                    data.build_batch(batch_data, context=False, context_oneline=True)
            elif config.model == "rnn":
                _, _, sentence, _, _, sentence_length = \
                    data.build_batch(batch_data, context=False, context_oneline=False)

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
            loss, _ = sess.run([seq2seq.loss, seq2seq.train_op], feed_dict=feed)
            total_loss += loss
            
            if batch % config.print_every_batch == 0:
                # 输出上次到这次的平均loss
                print_loss = total_loss if batch == 0 else total_loss / config.print_every_batch
                losses.append(print_loss)
                total_loss = 0
                print("-----------------------------")
                print("batch:", batch, "/", batches)
                print("time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                print("training loss:", print_loss)

            if batch % config.test_every_batch == 0 and batch > 0:
                generator_test = data.get_generator([structured_test], batch_size=config.batch_size)
                total_loss_test = 0
                for batch_test in range(batches_test):
                    batch_data_test = generator_test.__next__()
                    batch_data_test = batch_data_test[0]
                    if config.model == "cnn-brnn-rnn":
                        context_test, _, sentence_test, context_length_test, _, sentence_length_test = \
                            data.build_batch(batch_data_test, context=True, context_oneline=False)
                    elif config.model == "brnn-rnn":
                        _, context_oneline_test, sentence_test, _, context_oneline_length_test, sentence_length_test = \
                            data.build_batch(batch_data_test, context=False, context_oneline=True)
                    elif config.model == "rnn":
                        _, _, sentence_test, _, _, sentence_length_test = \
                            data.build_batch(batch_data_test, context=False, context_oneline=False)

                    if config.model == "cnn-brnn-rnn":
                        feed_test = {
                            cnn.context: context_test,
                            seq2seq.inputs_length: context_length_test,
                            seq2seq.targets: sentence_test,
                            seq2seq.targets_length: sentence_length_test,
                        }
                    elif config.model == "brnn-rnn":
                        feed_test = {
                            seq2seq.inputs: context_oneline_test,
                            seq2seq.inputs_length: context_oneline_length_test,
                            seq2seq.targets: sentence_test,
                            seq2seq.targets_length: sentence_length_test,
                        }
                    elif config.model == "rnn":
                        feed_test = {
                            seq2seq.targets: sentence_test,
                            seq2seq.targets_length: sentence_length_test,
                        }
                    loss_test, _ = sess.run([seq2seq.loss, seq2seq.train_op], feed_dict=feed_test)
                    total_loss_test += loss_test
                print("-----------------------------")
                print("test loss:", total_loss_test/batches_test)
                print(saver.save(sess, "checkpoint/"+config.model+"/model.ckpt", batch))
                test_losses.append(total_loss_test/batches_test)
        
        print(losses)
        print(test_losses)
