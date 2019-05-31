import os
import nltk


def buid_dict_file():
    word_to_id = {}
    dict_file = 'processed_files/word_to_id.txt'
    file1 = ['sentiment.train.0', 'sentiment.train.1',
             'sentiment.dev.0', 'sentiment.dev.1',
             'sentiment.test.0', 'sentiment.test.1']
    for file_item in file1:
        with open(file_item, 'r') as f:
            for item in f:
                item = item.strip()
                word_list = nltk.word_tokenize(item)
                # print(word_list)
                # input("===")
                for word in word_list:
                    word = word.lower()
                    if word not in word_to_id:
                        word_to_id[word] = 0
                    word_to_id[word] += 1
    file2 = ['reference.0', 'reference.1']
    for file_item in file2:
        with open(file_item, 'r') as f:
            for instance in f:
                instance = instance.strip()
                item1, item2 = instance.split('\t')
                for item in [item1, item2]:
                    word_list = nltk.word_tokenize(item)
                    # print(word_list)
                    # input("===")
                    for word in word_list:
                        word = word.lower()
                        if word not in word_to_id:
                            word_to_id[word] = 0
                        word_to_id[word] += 1
    print("Get word_dict success: %d words" % len(word_to_id))
    # write word_to_id to file
    word_dict_list = sorted(word_to_id.items(), key=lambda d: d[1], reverse=True)
    with open(dict_file, 'w') as f:
        f.write("<PAD>\n")
        f.write("<UNK>\n")
        f.write("<BOS>\n")
        f.write("<EOS>\n")
        for ii in word_dict_list:
            f.write("%s\t%d\n" % (str(ii[0]), ii[1]))
            # f.write("%s\n" % str(ii[0]))
    print("build dict finished!")
    return


def build_id_file():
    # load word_dict
    word_dict = {}
    num = 0
    with open('processed_files/word_to_id.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = line.strip()
            word = item.split('\t')[0]
            word_dict[word] = num
            num += 1
    print("Load embedding success! Num: %d" % len(word_dict))

    # generate id file
    file1 = ['sentiment.train.0', 'sentiment.train.1',
             'sentiment.dev.0', 'sentiment.dev.1',
             'sentiment.test.0', 'sentiment.test.1']
    for file_item in file1:
        id_file_data = []
        with open(file_item, 'r') as f:
            for item in f:
                item = item.strip()
                word_list = nltk.word_tokenize(item)
                # print(word_list)
                # input("===")
                id_list = []
                for word in word_list:
                    word = word.lower()
                    id = word_dict[word]
                    id_list.append(id)
                id_file_data.append(id_list)
        # write to file:
        with open("processed_files/%s" % file_item, 'w') as f:
            for item in id_file_data:
                f.write("%s\n" % (' '.join([str(k) for k in item])))

    file2 = ['reference.0', 'reference.1']
    for file_item in file2:
        id_file_data = []
        with open(file_item, 'r') as f:
            for instance in f:
                instance = instance.strip()
                item1, item2 = instance.split('\t')
                # 1
                word_list1 = nltk.word_tokenize(item1)
                id_list1 = []
                for word in word_list1:
                    word = word.lower()
                    id = word_dict[word]
                    id_list1.append(id)
                # 2
                word_list2 = nltk.word_tokenize(item2)
                id_list2 = []
                for word in word_list2:
                    word = word.lower()
                    id = word_dict[word]
                    id_list2.append(id)
                id_file_data.append([id_list1, id_list2])
        # write to file:
        with open("processed_files/%s" % file_item, 'w') as f:
            for item in id_file_data:
                f.write("%s\t%s\n" % (' '.join([str(k) for k in item[0]]), ' '.join([str(k) for k in item[1]])))
    print('build id file finished!')
    return


if __name__ == '__main__':
    buid_dict_file()
    build_id_file()

