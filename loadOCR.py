import pandas as pd
import numpy as np

num_states = 26

def load_ocr(ocr_data):
    print('========> Start Loading converted OCR')
    patterns_train = list()
    labels_train = list()
    patterns_test = list()
    labels_test = list()

    for entry in ocr_data:
        #print(entry['word_id'])
        pattern = dict()
        pattern['data'] = list()
        for pixel in entry['pixel']:
            pixel = np.append(pixel, [1])  ####???
            pattern['data'].append(pixel)
        pattern['data'] = np.array(pattern['data']).transpose()
        pattern['num_states'] = num_states

        label = entry['word']

        ### training fold = 0
        if entry['fold'] == 0:
            patterns_train.append(pattern)
            labels_train.append(label)
        else:
            patterns_test.append(pattern)
            labels_test.append(label)


        #print(patterns_train, labels_train)
        #break

    assert len(patterns_train) == len(labels_train)
    assert len(patterns_test) == len(labels_test)
    print('========> Done Loading converted OCR')
    return patterns_train, labels_train, patterns_test, labels_test


def convert_ocr(ocr_file):
    print('====> Start converting the raw ocr data')
    ocr_data = list()
    cur_word = dict()
    cur_word_id = -1
    with open(ocr_file) as f:
        for line in f:
            entry = line.strip().split('\t')
            idx = int(entry[0])
            letter = str(entry[1])
            next_id = int(entry[2])
            word_id = int(entry[3])
            position = int(entry[4])
            fold = int(entry[5])
            pij = entry[6:]
            pij = list(map(int, pij))

            pij = np.array(pij).reshape((16,8))
            pij = pij.transpose()
            pij = pij.reshape((-1,))

            #### starting a new word
            if cur_word_id < word_id:
                if cur_word_id >= 0:
                    ocr_data.append(cur_word.copy())
                cur_word['word'] = list()
                cur_word['pixel'] = list()
                cur_word['fold'] = fold
                cur_word['word_id'] = word_id
                cur_word_id = word_id
                #if word_id == 2:
                #    print(ocr_data)
                #    break

            cur_word['word'].append(ord(letter) - ord('a'))
            cur_word['pixel'].append(pij)
            assert cur_word['fold'] == fold
            assert cur_word['word_id'] == word_id

        ocr_data.append(cur_word.copy())

    print('====> Done converting the raw ocr data')
    return ocr_data

def loadOCRData(path):
    ocr_data = convert_ocr(path)
    return load_ocr(ocr_data)

if __name__ == '__main__':
    ocr = 'data/letter.data'
    ocr_data = convert_ocr(ocr)

    #print(ocr_data[-1])
    #print(len(ocr_data))  ### 6877
    patterns_train, labels_train, patterns_test, labels_test = load_ocr(ocr_data)

    print(len(labels_train))  ## 626
    print(len(labels_test))  ## 6251

    print(patterns_train[0], labels_train[0])
    print(patterns_train[-1], labels_train[-1])

    print(patterns_test[0], labels_test[0])
    print(patterns_test[-1], labels_test[-1])
