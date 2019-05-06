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
import utils
from configs import config, dirs

"""
Todo
- Analisar os filtros finais para ver o q é q eles consideram + positivo e + negativo ?
- Test
- Output backups with proper model information -> Allow multiple runs at the same time !!
- Randomize training set order -> Important? They've already divided into test/train/dev
- Not using dev anywhere

- Fix softmax warning
- Investigate .long() thing
- Investigate padding problem in transformPhrasesIntoSeqOfId
- Pass building embedding matrix to earlier stage to just perform once
- Dropout or Batch Norm ?
- check unused files

"""

pipeline = {
    'ProcessGlove': False,
    'ImportGlove': True,
    'ProcessStandford': False,  # fixme Ideally this should not e required, oly importing but i dont have time for this shit right now
    'ImportStandford': True,
    'Train': True,
}

def runTrainProgram(config, pipeline, extraOutputDir=None):

    if pipeline['ProcessGlove']:
        # Embeddings

        vectors = []
        words = []
        word2Id = {}
        idx = 1     # Must start at 1, so we can apply padding
        print("> Going through Glove file...")
        with open(config['embeddingsFile'], 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                words.append(word)
                word2Id[word] = idx
                idx += 1
                vect = np.array(line[1:]).astype(np.float)
                vectors.append(vect)


        vectors = bcolz.carray(vectors, rootdir=config['outputEmbeddings'], mode='w')
        vectors.flush()

        pickle.dump(words, open(config['outputDictionaryToID'], 'wb'))
        pickle.dump(word2Id, open(config['outputDictionary'], 'wb'))

        # Build embeddings matrix based on the dictionary
        print('> Building embeddings matrix in pytorch format')
        numKnownWords = len(vectors)
        embeddings = torch.zeros((numKnownWords, config['embeddingSize']))
        dictSize = 0

        # todo - Ideally this array would have every single word used in the training and testing sets
        for it, word in enumerate(words):
            if word in words:
                embeddings[it] = torch.from_numpy(vectors[word2Id[word]-1])
            else:
                # As long as the array above this is the words belonging to Glove, this will never run
                embeddings[it] = torch.from_numpy(np.random.normal(size=(config['embeddingSize'],)))

        torch.save(embeddings, config['outputEmbeddingsPytorchFormat'])

    if pipeline['ImportGlove']:
        print("> Importing processed glove ...")
        embeddings = torch.load(config['outputEmbeddingsPytorchFormat'])
        # print(embeddings.shape)

    if pipeline['ProcessStandford']:
        print("> Going through standford trees...")

        dataset = pytreebank.load_sst(dirs['trees'])
        trainingSet = dataset["train"]

        # This array is generated in processGlove
        word2Id = pickle.load(open(config['outputDictionary'], 'rb'))       # Indexed by word: wordId

        # Transform each phrase for a sequence of IDs and to torch format
        print("> Transforming phrases into sequence of IDs")
        trainingData, trainingLabels = utils.transformPhrasesIntoSeqOfId(trainingSet, word2Id)

        # todo not really important since i do it after the import but maybe i'm missing an unsqueeze here

        # Save for future usage
        torch.save(trainingData, config['outputTrainingSet'])
        # todo test-me:
        trainingLabels = torch.unsqueeze(trainingLabels, 1)
        torch.save(trainingLabels, config['outputTrainingSetLabels'])

    if pipeline['ImportStandford']:
        print("> Importing standford data into datasets format")
        trainingData = torch.load(config['outputTrainingSet'])
        trainingLabels = torch.load(config['outputTrainingSetLabels'])
        trainingLabels = torch.unsqueeze(trainingLabels, 1)

    if pipeline['Train']:

        if config['usePretrainedEmbeddings'] is False:
            embeddings = None

        print('> Initiating training')
        # fixme - trainingData still depends on glove and it shouldnt if we want to train the embeddings from source
        trainingData = torch.utils.data.DataLoader(trainingData, batch_size=config['batchSize'])

        model = Net(embedding_dim=config['embeddingSize'], knownEmbeddings=embeddings,
                    layersConfig=config['modelConfig'], hiddenSize=config['hiddenSize'], dropOutRate=config['dropOutRatio']).to(config['device'])

        optimizer = optim.SGD(model.parameters(), lr=config['learningRate'], momentum=config['momentum'],
                              weight_decay=config['weightDecay'])

        modelBackupsIterator = 0

        for epoch in range(1, config['numEpochs'] + 1):
            model.train()
            loss_list = []
            acc_list = []
            print('- Epoch ' + str(epoch))
            # for batch_idx, (trainInstance, trainLabel) in enumerate(trainingData):
            for batch_idx, trainInstance in enumerate(trainingData):
                # fixme, ignoring last few pieces of training data that dont form an entire batch
                if len(trainingLabels) < batch_idx*config['batchSize']+config['batchSize']:
                    break

                batchLabels = trainingLabels[batch_idx*config['batchSize']: batch_idx*config['batchSize']+config['batchSize']]
                data, target = trainInstance.to(config['device']), batchLabels.to(config['device'])
                target = target.view((config['batchSize'],))
                optimizer.zero_grad()
                output = model(data.long())
                # todo, the training instances should be already saved as 'long' instead of changed here
                # todo, also investigate if this does not affect the results
                # print(output.shape)
                # print(target.shape)

                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                # print("===== TARG ===")
                # print(target)
                # print(target.shape)
                # print("===== PRED ===")
                # print(pred)
                # print(pred.shape)
                acc = pred.eq(target.view_as(pred)).float().mean()
                acc_list.append(acc.item())
                if batch_idx % config['logInterval'] == 0:
                    msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tAvg Loss: {:.4f}\tAvg Acc: {:.4f}'.format(
                        epoch, batch_idx * config['batchSize'], len(trainingData.dataset),
                               100*batch_idx*config['batchSize'] / len(trainingData.dataset), np.mean(loss_list), np.mean(acc_list))
                    print(msg)
                    if extraOutputDir is not None:
                        torch.save(pred, extraOutputDir + 'epoch_' + str(epoch) + '_' + str(modelBackupsIterator) + '.pt')

                    loss_list.clear()
                    acc_list.clear()

                if batch_idx % config['modelBackupInterval'] == 0:
                    if batch_idx is not 0:
                        print('> Backup at batch ID {} for the {}th time'.format(str(batch_idx), str(modelBackupsIterator)))
                        torch.save(model.state_dict(), config['outputModelBackup'] +
                                   '_epoch_' + str(epoch) + '_' + str(modelBackupsIterator) + '.pt')
                        modelBackupsIterator += 1

        torch.save(model.state_dict(), config['outputFinalModel'])


# runTrainProgram(config, pipeline)