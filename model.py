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
        aux = torch.Tensor()
        for it1, trainingInst in enumerate(x):
            for it2, wID in enumerate(trainingInst):
                aux = torch.cat((aux, x[it1, it2]))
            new_x[it1, 0] = aux
        return new_x

# todo
# batch normalization ?
# dropout ?
class Net(nn.Module):
    def __init__(self, embedding_dim=300, knownEmbeddings=[]):
        super(Net, self).__init__()
        self.embeddings = initEmbeddingLayer(knownEmbeddings, False)
        self.concatEmbed = concat()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool1d(embedding_dim),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool1d(embedding_dim * 5),
        )

        self.lin2 = nn.Linear(128, 5)

    def forward(self, x):
        # print("> Forward embed:")
        out = self.embeddings(x)
        # print("> Concatenate layer:")
        out = self.concatEmbed(out)
        # print("> Forward layer 1:")
        out = self.layer1(out)
        # print("> Forward layer 2:")
        out = self.layer2(out)
        # print("> Forward reshape:")
        out = out.reshape(out.size(0), -1)
        # print("> Forward softmax:")
        out = F.softmax(self.lin2(out))
        # print("> Forward done")
        return out

