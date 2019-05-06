import torch
import xlsxwriter

def transformPhrasesIntoSeqOfId(set, word2Id):
    newSet = []
    setLabelsAux = []
    nonIgnored = 0
    ignored = 0
    biggestPhraseLen = 0
    origNumOfPhrases = 0

    for phraseSet in set:
        phraseSet.lowercase()  # to lowercase
        #print(phraseSet)
        """
        # Initial state where we used every setence combinaation for every phrase - turne out overfitting the model
        for it, (label, sentence) in enumerate(phraseSet.to_labeled_lines()):

            '''print("%s has sentiment label %s" % (
                sentence,
                ["very negative", "negative", "neutral", "positive", "very positive"][label]
            ))'''
            origNumOfPhrases += 1
            newSentence = []
            for w in sentence.split():  # todo splitting by spaces not sure if best choice
                if w in word2Id:
                    newSentence.append(word2Id[w])
                    nonIgnored += 1
                else:
                    ignored += 1
            if len(newSentence) is not 0:
                newSet.append(newSentence)
                setLabelsAux.append(label)
                if len(newSentence) > biggestPhraseLen:
                    biggestPhraseLen = len(newSentence)
        """
        label, sentence = phraseSet.to_labeled_lines()[0]
        origNumOfPhrases += 1
        newSentence = []
        for w in sentence.split():  # todo splitting by spaces not sure if best choice
            if w in word2Id:
                newSentence.append(word2Id[w])
                nonIgnored += 1
            else:
                ignored += 1
        if len(newSentence) is not 0:
            newSet.append(newSentence)
            setLabelsAux.append(label)
            if len(newSentence) > biggestPhraseLen:
                biggestPhraseLen = len(newSentence)

    print('- Number of phrases went from ' + str(origNumOfPhrases) + ' to ' + str(len(newSet)))
    print('- Ignored ' + str(ignored) + ' words (out of dictionary).')
    print('- Recognized ' + str(nonIgnored) + ' words.')
    print('- Biggest phrase has ' + str(biggestPhraseLen) + ' words.')

    # Transform both sets to torch format
    newSetTorchForm = torch.zeros(len(newSet), biggestPhraseLen, dtype=torch.int32)
    for it, data in enumerate(newSet):
        paddedData = (data + biggestPhraseLen*[0])[:biggestPhraseLen]        # Only adding zeros to the end of the arr
        # Careful because there's one word which already has ID 0
        newSetTorchForm[it] = torch.Tensor(paddedData)

    setLabelsTorchForm = torch.tensor(setLabelsAux)

    assert(newSetTorchForm.shape[0] == setLabelsTorchForm.shape[0])

    return newSetTorchForm, setLabelsTorchForm

def log(outputPath, config, configWantedKeys,specificChanges, loss, acc):
    writeDictToExel(outputPath + '_fullConf', config, configWantedKeys)
    writeDictToExel(outputPath + '_SpecificConf', specificChanges, specificChanges.keys())

    f = open('loss= ' + loss + ' acc= ' + acc + ' .txt', "w")
    f.write('loss= ' + loss + ' acc= ' + acc)
    f.close()


def writeDictToExel(filename, d, keys):

    workbook = xlsxwriter.Workbook(filename + '.xlsx')
    worksheet = workbook.add_worksheet()

    row = 0
    col = 0

    for key in keys:
        row += 1
        worksheet.write(row, col, key)
        worksheet.write(row, col + 1, d[key])

    workbook.close()