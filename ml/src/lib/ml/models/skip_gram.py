import torch
import torch.nn as nn


class SkipGram(nn.Module):
    """Skip-gram model as used in NLP

    Shape of output: (*target.shape, contextSize, vocabularySize)
    """

    def __init__(self, vocabularySize: int, embeddingSize: int, contextSize: int = 1):
        super().__init__()
        self.embeddingLayer = nn.Embedding(vocabularySize, embeddingSize)
        self.outputLayers   = nn.ModuleList(nn.Linear(embeddingSize, vocabularySize) for _ in range(contextSize))

    def forward(self, target: torch.Tensor):
        embedding = self.embeddingLayer(target)
        outputs   = [outputLayer(embedding) for outputLayer in self.outputLayers]
        return torch.stack(outputs, dim=-2)

    @property
    def embeddings(self):
        return self.embeddingLayer.weight

    @property
    def embeddingsKey(self):
        return "embeddingLayer.weight"
