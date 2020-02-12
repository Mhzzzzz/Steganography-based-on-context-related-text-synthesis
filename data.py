"""padding, 分割训练和测试集"""

import numpy as np
import random


def get_generator(inputs, batch_size=None, shuffle=False):
    """
        循环产生批量数据batch
        :param inputs: list类型数据，多个list,请[list0,list1,...]
        :param batch_size: batch大小
        :param shuffle: 是否打乱inputs数据
        :return: 返回一个batch数据
    """
    rows = len(inputs[0])
    indices = list(range(rows))
    if shuffle:
        random.shuffle(indices)
    while True:
        batch_indices = np.asarray(indices[0:batch_size])  # 产生一个batch的index
        indices = indices[batch_size:] + indices[:batch_size]  # 循环移位，以便产生下一个batch
        batch_data = []
        for data in inputs:
            data = np.asarray(data)
            temp_data = data[batch_indices]  # 使用下标查找，必须是ndarray类型类型
            batch_data.append(temp_data.tolist())
        yield batch_data


def build_batch(structured_list, context=True, context_oneline=False):
    batch_context = []
    batch_context_oneline = []
    batch_sentence = []
    batch_context_length = []
    batch_context_oneline_length = []
    batch_sentence_length = []
    for structure_dict in structured_list:
        if context:
            batch_context.append(structure_dict["context"])
            batch_context_length.append(structure_dict["context_length"])
        if context_oneline:
            batch_context_oneline.append(structure_dict["context_oneline"])
            batch_context_oneline_length.append(structure_dict["context_oneline_length"])
        batch_sentence.append(structure_dict["sentence"])
        batch_sentence_length.append(structure_dict["sentence_length"])
    return batch_context, batch_context_oneline, batch_sentence, \
           batch_context_length, batch_context_oneline_length, batch_sentence_length


"""
def process_bert(config, context_min=1):
    data = np.load("dataset/" + config.dataset + "/data.npy", allow_pickle=True)
    w2i = np.load("dataset/" + config.dataset + "/w2i.npy", allow_pickle=True).item()
    i2w = np.load("dataset/" + config.dataset + "/i2w.npy", allow_pickle=True).item()

    # build dictionary
    structured_list = []
    for story in data:
        for i in range(context_min, len(story)):
            structured_dict = {}
            structured_dict["context"] = []
            structured_dict["sentence"] = []
            for j in range(i):
                story_word = [i2w[x] for x in story[j]]
                structured_dict["context"].append(" ".join(story_word))
            structured_dict["context_length"] = i
            structured_dict["sentence"] = [_ for _ in story[i]]
            structured_dict["sentence_length"] = len(story[i]) + 1  # all words + _eos
            structured_list.append(structured_dict)

    # compute max
    context_length = []
    sentence_length = []
    for d in structured_list:
        context_length.append(d["context_length"])
        sentence_length.append(d["sentence_length"])
    config.max_context_length = max(context_length)
    config.max_sentence_length = max(sentence_length)

    # padding
    for d in structured_list:
        for i in range(config.max_context_length - d["context_length"]):
            d["context"].append("")
        d["sentence"] += [w2i["_EOS"]] + [w2i["_PAD"]]*(config.max_sentence_length - d["sentence_length"])

    # cut
    total_num = len(structured_list)
    train_num = int(total_num * config.ratio)
    structured_train = structured_list[:train_num]
    structured_test = structured_list[train_num:]

    return structured_train, structured_test, w2i, i2w
"""


def build_dictionary(data, context_min, need_context=True):
    structured_list = []
    FLAG = 0
    if context_min == -1:
        FLAG = 1
    for story in data:
        if FLAG:
            context_min = len(story) - 1
        for i in range(context_min, len(story)):
            structured_dict = {}
            structured_dict["context"] = []
            structured_dict["sentence"] = []
            if need_context:
                for j in range(i):
                    structured_dict["context"].append([_ for _ in story[j]])
            structured_dict["context_length"] = i
            structured_dict["sentence"] = [_ for _ in story[i]]
            structured_dict["sentence_length"] = len(story[i]) + 1  # all words + _eos
            structured_list.append(structured_dict)
    return structured_list


def combine_context(structured_list):
    for d in structured_list:
        d["context_oneline"] = []
        for i in range(d["context_length"]):
            d["context_oneline"] += [_ for _ in d["context"][i]]
        d["context_oneline_length"] = len(d["context_oneline"]) + 1


def compute_max(config, structured_list, all_sentences=True, context=True, context_oneline=False):
    if context:
        context_length = []
        for d in structured_list:
            context_length.append(d["context_length"])
        config.max_context_length = max(context_length)

    if context_oneline:
        context_oneline_length = []
        for d in structured_list:
            context_oneline_length.append(d["context_oneline_length"])
        config.max_context_oneline_length = max(context_oneline_length)

    sentence_length = []
    for d in structured_list:
        sentence_length.append(d["sentence_length"])
        if all_sentences:
            for i in range(d["context_length"]):
                sentence_length.append(len(d["context"][i]) + 1)
    config.max_sentence_length = max(sentence_length)


def padding(config, structured_list, w2i, context=True, context_oneline=False):
    for d in structured_list:
        d["sentence"] += [w2i["_EOS"]] + [w2i["_PAD"]]*(config.max_sentence_length - d["sentence_length"])

    if context:
        for d in structured_list:
            for i in range(d["context_length"]):
                d["context"][i] += [w2i["_EOS"]] + [w2i["_PAD"]] * (config.max_sentence_length - len(d["context"][i]) - 1)
            for i in range(config.max_context_length - d["context_length"]):
                d["context"].append([w2i["_PAD"]] * config.max_sentence_length)

    if context_oneline:
        for d in structured_list:
            d["context_oneline"] += [w2i["_EOS"]] + [w2i["_PAD"]]*(config.max_context_oneline_length - d["context_oneline_length"])


def process(config, context_min=1, need_context=True, context_oneline=False):
    data = np.load("dataset/" + config.dataset + "/data.npy", allow_pickle=True)
    w2i = np.load("dataset/" + config.dataset + "/w2i.npy", allow_pickle=True).item()
    i2w = np.load("dataset/" + config.dataset + "/i2w.npy", allow_pickle=True).item()

    # build dictionary
    structured_list = build_dictionary(data, context_min, need_context)
    if context_oneline:
        combine_context(structured_list)

    # compute max and padding
    if need_context:
        if context_oneline:
            compute_max(config, structured_list, all_sentences=False, context=False, context_oneline=True)
            padding(config, structured_list, w2i, context=False, context_oneline=True)
        else:
            compute_max(config, structured_list, all_sentences=True, context=True, context_oneline=False)
            padding(config, structured_list, w2i, context=True, context_oneline=False)
    else:
        compute_max(config, structured_list, all_sentences=False, context=False, context_oneline=False)
        padding(config, structured_list, w2i, context=False, context_oneline=False)

    # cut
    total_num = len(structured_list)
    train_num = int(total_num * config.ratio)
    structured_train = structured_list[:train_num]
    structured_test = structured_list[train_num:]

    return structured_train, structured_test, w2i, i2w


if __name__ == "__main__":
    import configs
    config = configs.Config()
    structured_train, structured_test, w2i, i2w = process(config, need_context=False, oneline_context=False)
    generator = get_generator([structured_train], batch_size=config.batch_size, shuffle=True)
    batch = generator.__next__()
    batch = batch[0]
    context, sentence, context_length, sentence_length = build_batch(batch)
    pass
