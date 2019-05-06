import os
import torch

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

dirs = {
    'wordVectors': 'data',
    'standford': os.path.join('data', 'standfordSentimentTreebank'),
    'trees': os.path.join('data', 'trees'),
    'gloveOutput': 'gloveOutput',
    # 'standfordOutput': 'trashDirectory',
    'standfordOutput': 'standfordOutput',
    'modelOutput': 'modelOutput',
    # 'modelOutput': 'modelSecondOutput',
}

config = {
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

    'embeddingSize': 300,
    'batchSize': 100,
    'numEpochs': 3,
    'learningRate': 0.01,
    'momentum': 0.9,
    'weightDecay': 0.001,
    'hiddenSize': 256,
    'modelConfig': [{
            'inChan': 1,
            'outChan': 128,
            'kernSiz': 3,
        }, {
            'inChan': 128,
            'outChan': 128,
            'kernSiz': 3,
        }],

    'logInterval': 50,
    'modelBackupInterval': 500,
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),

}
