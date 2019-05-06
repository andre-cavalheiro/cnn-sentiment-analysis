from configs import config, dirs
from test import runTestProgram
from train import runTrainProgram, pipeline
from utils import log
import string
import random
import os

configLogKeys = [
    'embeddingSize',
    'batchSize',
    'numEpochs',
    'learningRate',
    'momentum',
    'weightDecay',
    'hiddenSize',
    'usePretrainedEmbeddings',
    'dropOutRatio',
    'modelConfig'
]

diffConfsToTest = [
    {
        'dropOutRatio': 0.2
    },
    {
        'hiddenSize': 516
    },
    {
        'dropOutRatio': 0.2,
        'hiddenSize': 516
    },
    {
        'modelConfig': [{
            'inChan': 1,
            'outChan': 128,
            'kernSiz': 2,
        }, {
            'inChan': 128,
            'outChan': 128,
            'kernSiz': 2,
        }],
    },
    {
        'modelConfig': [{
            'inChan': 1,
            'outChan': 128,
            'kernSiz': 2,
        }, {
            'inChan': 128,
            'outChan': 128,
            'kernSiz': 2,
        }],
        'dropOutRatio': 0.2
    },
    {
        'modelConfig': [{
            'inChan': 1,
            'outChan': 256,
            'kernSiz': 3,
        }, {
            'inChan': 256,
            'outChan': 256,
            'kernSiz': 3,
        }],
    },
    {
        'modelConfig': [{
            'inChan': 1,
            'outChan': 256,
            'kernSiz': 3,
        }, {
            'inChan': 256,
            'outChan': 256,
            'kernSiz': 3,
        }],
        'dropOutRatio': 0.2
    },
    {
        'modelConfig': [{
            'inChan': 1,
            'outChan': 128,
            'kernSiz': 2,
        }, {
            'inChan': 128,
            'outChan': 128,
            'kernSiz': 3,
        }],
    },
    {
        'modelConfig': [{
            'inChan': 1,
            'outChan': 128,
            'kernSiz': 2,
        }, {
            'inChan': 128,
            'outChan': 128,
            'kernSiz': 3,
        }],
        'dropOutRatio': 0.2
    },
    {
        'modelConfig': [{
            'inChan': 1,
            'outChan': 128,
            'kernSiz': 2,
        }, {
            'inChan': 128,
            'outChan': 128,
            'kernSiz': 3,
        }],
        'hiddenSize': 516
    },
    {
        'modelConfig': [{
            'inChan': 1,
            'outChan': 128,
            'kernSiz': 2,
        }, {
            'inChan': 128,
            'outChan': 128,
            'kernSiz': 3,
        }],
        'hiddenSize': 516,
        'dropOutRatio': 0.2
    }
]

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
   return ''.join(random.choice(chars) for _ in range(size))

configId = []

for specificConf in diffConfsToTest:
    conf_ = config.copy()
    for k in specificConf.keys():
        conf_[k] = specificConf[k]
    id = id_generator()

    print("== [CONFIG ID]: " + id + " ==")
    outputPath = os.path.join('configuration_'+id, dirs['modelOutput'])
    os.makedirs(outputPath)
    conf_['outputModelBackup'] = os.path.join(outputPath, 'modelBackup')
    conf_['outputFinalModel'] = os.path.join(outputPath, 'finalModel.pt')
    configId.append(id)

    print("== Folders generated, initiating training ==")
    runTrainProgram(conf_, pipeline, extraOutputDir=outputPath)

    print("== Initiating tests ==")
    loss, acc = runTestProgram(conf_)      # fixme, still not working when embeddings are built from scratch

    log(outputPath, conf_, configLogKeys, specificConf, loss, acc)

f = open('ConfigurationIndexes.txt', "w")
for i, c in enumerate(configId):
    f.write(i + ' - ' + c)
f.close()
