from configs import config, dirs
from model import Net

import pytreebank
import pickle

import torch
import torchvision
import torch.nn.functional as F

import utils
from model import Net

def runTestProgram(config):
    print("> Importing testing Data")
    word2Id = pickle.load(open(config['outputDictionary'], 'rb'))  # Indexed by word: wordId
    dataset = pytreebank.load_sst(dirs['trees'])
    testingSet = dataset["test"]

    # Transform each phrase for a sequence of IDs
    print("> Transforming phrases into sequence of IDs")
    testData, testLabels = utils.transformPhrasesIntoSeqOfId(testingSet, word2Id)
    testLabels = torch.unsqueeze(testLabels, 1)

    if config['usePretrainedEmbeddings'] is False:
        # fixme
        embeddings = None
    else:
        print("> Importing Embeddings matrix")
        embeddings = torch.load(config['outputEmbeddingsPytorchFormat'])

    print("> Importing Model")
    model = Net(embedding_dim=config['embeddingSize'], knownEmbeddings=embeddings,
                    layersConfig=config['modelConfig'], hiddenSize=config['hiddenSize']).to(config['device'])

    model.load_state_dict(torch.load(config['outputFinalModel']))

    print('> Initiating testing')
    testData = torch.utils.data.DataLoader(testData, batch_size=config['batchSize'])
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0

        # for batch_idx, (trainInstance, trainLabel) in enumerate(trainingData):
        for batch_idx, trainInstance in enumerate(testData):
            # fixme, ignoring last few pieces of training data
            if len(testLabels) < batch_idx * config['batchSize'] + config['batchSize']:
                break

            batchLabels = testLabels[
                          batch_idx * config['batchSize']: batch_idx * config['batchSize'] + config['batchSize']]
            data, target = trainInstance.to(config['device']), batchLabels.to(config['device'])
            target = target.view((config['batchSize'],))

            output = model(data.long())

            test_loss += F.cross_entropy(output, target).item()  # sum up batch loss

            pred = output.argmax(dim=1, keepdim=True)            # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            if batch_idx % config['logTestInterval'] == 0:
                print('[{}/{} ({:.0f}%)]'.format(batch_idx * config['batchSize'], len(testData.dataset),
                                   100*batch_idx*config['batchSize'] / len(testData.dataset)))

    test_loss /= len(testData.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(testData.dataset),
        100. * correct / len(testData.dataset)))

    return test_loss, 100 * correct / len(testData.dataset)

# runTestProgram(config)