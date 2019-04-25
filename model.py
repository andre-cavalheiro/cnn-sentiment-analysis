import torch
import torch.nn as nn
import torch.nn.functional as F


def initEmbeddingLayer(embeddings, trainable):
    numEmbed = embeddings.shape[0]
    dimEmbed = embeddings.shape[1]
    print('> Initializing embedding layer for ' + str(numEmbed) + ' embeddings with size ' + str(dimEmbed))

    layer = nn.Embedding(num_embeddings=numEmbed, embedding_dim=dimEmbed)
    layer.load_state_dict({'weight': embeddings})  # Load known embeddings
    if not trainable:
        layer.weight.requires_grad = False

    return layer

class concat(nn.Module):
    def __init__(self):
        super(concat, self).__init__()

    def forward(self, x):
        new_x = torch.Tensor(x.shape[0], 1, x.shape[1]*x.shape[2])  # 1 because num_in_chanels
        for it1, trainingInst in enumerate(x):
            aux = torch.Tensor()
            for it2, wID in enumerate(trainingInst):
                aux = torch.cat((aux, x[it1, it2]))
            new_x[it1, 0] = aux
        return new_x

class Net(nn.Module):
    def __init__(self, embedding_dim=300, knownEmbeddings=[], layersConfig=[]):
        super(Net, self).__init__()
        self.embeddings = initEmbeddingLayer(knownEmbeddings, False)
        self.concatEmbed = concat()
        self.layers = []
        mul = 1
        for it, conf in enumerate(layersConfig):
            if it is len(layersConfig)-1:
                mul = 5                     # todo What exactly is this doing?

            self.layers.append(nn.Sequential(
                nn.Conv1d(conf['inChan'], conf['outChan'], kernel_size=conf['kernSiz']),
                nn.ReLU(),
                nn.MaxPool1d(embedding_dim*mul),
            ))
        self.lin = nn.Linear(layersConfig[len(layersConfig)-1]['outChan'], 5)  # Number of sentiments(classes) possible

    def forward(self, x):
        # print("> Forward embed:")
        out = self.embeddings(x)
        # print("> Concatenate layer:")
        out = self.concatEmbed(out)
        # print("> Forward layers:")
        for layer in self.layers:
            out = layer(out)
        # print("> Forward reshape:")
        out = out.reshape(out.size(0), -1)
        # print("> Forward softmax:")
        out = F.softmax(self.lin(out))
        # print("> Forward done")
        return out



