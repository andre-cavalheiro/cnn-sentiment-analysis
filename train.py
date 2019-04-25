import os
import pytreebank
import bcolz
import numpy as np
import pickle

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import Net

# todo


"""
    Standford sentiment tree bank:
        Note difference between sentenceId and phraseId
        File description:
            - datasetSentences: sentenceId - sentence       -> sentences from train/test/dev sets
            - datasetSplit: sentenceId - set (1=train, 2=test, 3=dev)
            
            - dictionary: phrase | phraseId 
            - sentiment_labels: phraseId - sentimentValue(0-1)
    Trees:
        train/test/dev sets in tree format
    Glove: 
        word wordEmbedding,          
"""

pipeline = {
    'ProcessGlove': False,
    'ImportGlove': True,
    'ProcessStandford': False,
    'ImportStandford': True,
    'Train': True,
}

dirs = {
    'wordVectors': 'data',
    'standford': os.path.join('data', 'standfordSentimentTreebank'),
    'trees': os.path.join('data', 'trees'),
    'gloveOutput': 'gloveOutput',
    'standfordOutput': 'standfordOutput',
    'modelOutput': 'modelOutput',
}

config = {
    'embeddingSize': 300,
    'embeddingsFile': os.path.join(dirs['wordVectors'], 'glove.6B.300d.txt'),
    'tree': os.path.join(dirs['standford'], 'datasetSentences'),
    'outputEmbeddings': os.path.join(dirs['gloveOutput'], 'embeddings.dat'),
    'outputEmbeddingsPytorchFormat': os.path.join(dirs['gloveOutput'], 'embeddingsPT.pt'),
    'outputDictionary': os.path.join(dirs['gloveOutput'], 'dict.pkl'),
    'outputDictionaryToID': os.path.join(dirs['gloveOutput'], 'dictToId.pkl'),
    'outputTrainingSet': os.path.join(dirs['standfordOutput'], 'train.pt'),
    'outputTrainingSetLabels': os.path.join(dirs['standfordOutput'], 'trainLabels.pt'),
    'outputModelBackup': os.path.join(dirs['modelOutput'], 'modelBackup'),
    'outputFinalModel': os.path.join(dirs['modelOutput'], 'finalModel.pt'),

    'batchSize': 100,
    'numEpochs': 3,
    'logInterval': 50,
    'modelBackupInterval': 10000,
    'learningRate': 0.01,
    'momentum': 0.9,
    'weightDecay': 0.001,
    'modelConfig': [{
        'inChan': 1,
        'outChan': 128,
        'kernSiz': 2,
    }, {
        'inChan': 128,
        'outChan': 128,
        'kernSiz': 2,
    }],

    # todo
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'kwargs': {'num_workers': 1, 'pin_memory': True},
}


if pipeline['ProcessGlove']:
    # Embeddings

    vectors = []
    words = []
    word2Id = {}
    idx = 0
    print("> Going through Glove file...")
    with open(config['embeddingsFile'], 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2Id[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            # assert(len(vect) == config['embeddingSize'])
            vectors.append(vect)

    """print("> Storing data")
    print(vectors[0])
    print(vectors[1])"""

    vectors = bcolz.carray(vectors, rootdir=config['outputEmbeddings'], mode='w')
    vectors.flush()

    pickle.dump(words, open(config['outputDictionaryToID'], 'wb'))
    pickle.dump(word2Id, open(config['outputDictionary'], 'wb'))

if pipeline['ImportGlove']:
    # Import
    print("> Importing processed glove ...")

    vectors = bcolz.open(config['outputEmbeddings'])[:]
    words = pickle.load(open(config['outputDictionaryToID'], 'rb'))     # Indexed by wordId: word
    word2Id = pickle.load(open(config['outputDictionary'], 'rb'))       # Indexed by word: wordId
    assert(len(vectors) == len(words))
    glove = {w: vectors[word2Id[w]] for w in words}                     # Indexed by word: embedding
    # todo pretty sure we can delete the previous 3 vectors

    # print(vectors[:2])
    # print(words[:10])
    # print(word2Id[words[0]])
    # print(glove[words[0]])

if pipeline['ProcessStandford']:
    print("> Going through standford trees...")

    # Train, Test, Dev datasets
    dataset = pytreebank.load_sst(dirs['trees'])
    trainingSet = dataset["train"]

    """
        # Nº de linhas nos respetivos ficheiros
        example = dataset["train"][0]

        print(len(dataset["train"]))                # 8544 
        print(len(dataset["test"]))                 # 2210
        print(len(dataset["dev"]))                  # 1101

        # Possiveis combinações das palavras nas frases? Is that it?

        print(len(dataset["train"][0].to_lines()))  # 71
        print(len(dataset["train"][1].to_lines()))  # 73
        print(len(dataset["train"][2].to_lines()))  # 77

        print(len(dataset["test"][0].to_lines()))   # 7
        print(len(dataset["test"][1].to_lines()))   # 41
        print(len(dataset["test"][2].to_lines()))   # 45

        print(len(dataset["dev"][0].to_lines()))    # 25
        print(len(dataset["dev"][1].to_lines()))    # 25
        print(len(dataset["dev"][2].to_lines()))    # 47
        """

    # Transform each phrase for a sequence of IDs
    print("> Transforming phrases into sequence of IDs")
    newTrainingSet = []
    nonIgnored = 0
    ignored = 0
    biggestPhraseLen = 0
    origNumOfPhrases = 0
    trainingLabelsAux = []
    for phraseSet in trainingSet:
        phraseSet.lowercase()   # to lowercase
        for it, (label, sentence) in enumerate(phraseSet.to_labeled_lines()):

            """print("%s has sentiment label %s" % (
                sentence,
                ["very negative", "negative", "neutral", "positive", "very positive"][label]
            ))"""
            origNumOfPhrases += 1
            newSentence = []
            for w in sentence.split():  # todo splitting by spaces not sure if best choice
                if w in word2Id:
                    newSentence.append(word2Id[w])
                    nonIgnored += 1
                else:
                    ignored += 1
            if len(newSentence) is not 0:
                newTrainingSet.append(newSentence)
                trainingLabelsAux.append(label)

                if len(newSentence) > biggestPhraseLen:
                    biggestPhraseLen = len(newSentence)
        """
        if(len(newTrainingSet)>3):
            break
        """
    print('- Number of phrases went from ' + str(origNumOfPhrases) + ' to ' + str(len(newTrainingSet)))
    print('- Ignored ' + str(ignored) + ' words (out of dictionary).')
    print('- Recognized ' + str(nonIgnored) + ' words.')
    print('- Biggest phrase has ' + str(biggestPhraseLen) + ' words.')

    """print(newTrainingSet[0])
    print(newTrainingSet[1])
    print(newTrainingSet[2])
    i=0
    for label, sentence in trainingSet[0].to_labeled_lines():
        print(sentence)
        i+=1
        if(i>=3):
            break
    """

    # Transform to torch format
    trainingLabels = torch.tensor(trainingLabelsAux)
    trainingData = torch.zeros(len(newTrainingSet), biggestPhraseLen, dtype=torch.int32)
    for it, data in enumerate(newTrainingSet):
        paddedData = (data + biggestPhraseLen*[0])[:biggestPhraseLen]        # Only adding zeros to the end of the arr
        # Careful because there's one word which already has ID 0
        trainingData[it] = torch.Tensor(paddedData)

    assert(trainingData.shape[0] == trainingLabels.shape[0])

    # Save for future usage
    torch.save(trainingData, config['outputTrainingSet'])
    torch.save(trainingLabels, config['outputTrainingSetLabels'])

if pipeline['ImportStandford']:
    print("> Importing standford data into datasets format")
    trainingData = torch.load(config['outputTrainingSet'])
    trainingLabels = torch.load(config['outputTrainingSetLabels'])
    trainingLabels = torch.unsqueeze(trainingLabels, 1)

if pipeline['Train']:
    # Build embeddings matrix based on the dictionary
    # fixme - this can be put into ProcessGlove
    # fixme - Dont know where the dictionary is yet, this is just converting from NP to torch

    print('> Building embeddings matrix in pytorch format')
    numKnownWords = len(vectors)
    embeddings = torch.zeros((numKnownWords, config['embeddingSize']))
    dictSize = 0

    for it, word in enumerate(words):
        # it will match wordId
        if word in glove.keys():
            embeddings[it] = torch.from_numpy(glove[word])
        else:
            embeddings[it] = torch.from_numpy(np.random.normal(size=(config['embeddingSize'],)))

    torch.save(embeddings, config['outputEmbeddingsPytorchFormat'])

    print('> Initiating training')
    trainingData = torch.utils.data.DataLoader(trainingData, batch_size=config['batchSize'])
    # trainingData_ = torch.utils.data.TensorDataset(trainingData, trainingLabels)

    model = Net(embedding_dim=config['embeddingSize'], knownEmbeddings=embeddings,
                layersConfig=config['modelConfig']).to(config['device'])

    optimizer = optim.SGD(model.parameters(), lr=config['learningRate'], momentum=config['momentum'],
                          weight_decay=config['weightDecay'])

    modelBackupsIterator = 0
    for epoch in range(1, config['numEpochs'] + 1):
        model.train()
        loss_list = []
        acc_list = []
        print('> Epoch ' + str(epoch))
        # for batch_idx, (trainInstance, trainLabel) in enumerate(trainingData):
        for batch_idx, trainInstance in enumerate(trainingData):
            batchLabels = trainingLabels[batch_idx*config['batchSize']: batch_idx*config['batchSize']+config['batchSize']]
            data, target = trainInstance.to(config['device']), batchLabels.to(config['device'])
            target = target.view((len(target),))
            optimizer.zero_grad()
            output = model(data.long())
            # todo, the training instances should be already saved as long instead of changed here
            # todo, also investigate if this does not affect the results
            # print(output.shape)
            # print(target.shape)

            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            acc = pred.eq(target.view_as(pred)).float().mean()
            acc_list.append(acc.item())
            if batch_idx % config['logInterval'] == 0:
                msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tAvg Loss: {:.4f}\tAvg Acc: {:.4f}'.format(
                    epoch, batch_idx * config['batchSize'], len(trainingData.dataset),
                           100 * batch_idx / len(trainingData.dataset), np.mean(loss_list), np.mean(acc_list))
                print(msg)
                loss_list.clear()
                acc_list.clear()

            if batch_idx % config['logInterval'] == 0:
                torch.save(model.state_dict(), config['outputModelBackup'] + str(modelBackupsIterator) + '.pt')
                modelBackupsIterator += 1
            """
            to import:
                the_model = TheModelClass(*args, **kwargs)
                the_model.load_state_dict(torch.load(PATH))
            """
    torch.save(model.state_dict(), config['outputFinalModel'])
