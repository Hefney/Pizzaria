import torch
from torch import nn


class NERDataset(torch.utils.data.Dataset):

    def __init__(self, x, y, pad, tag_none):
        """
        This is the constructor of the NERDataset
        Inputs:
        - x: a list of lists where each list contains the ids of the tokens
        - y: a list of lists where each list contains the label of each token in the sentence
        - pad: the id of the <PAD> token (to be used for padding all sentences and labels to have the same length)
        """
        x_max = max(x, key=lambda z: len(z))
        x = [sentence + [pad] * (len(x_max) - len(sentence)) for sentence in x]
        y_max = max(y, key=lambda z: len(z))
        y = [sentence + [tag_none] * (len(y_max) - len(sentence)) for sentence in y]
        self.x_tensor = torch.tensor(x)
        self.y_tensor = torch.tensor(y)

    def __len__(self):
        """
        This function should return the length of the dataset (the number of sentences)
        """
        return self.x_tensor.shape[0]

    def __getitem__(self, idx):
        """
        This function returns a subset of the whole dataset
        """
        return self.x_tensor[idx], self.y_tensor[idx]


class NER(nn.Module):
    def __init__(self, vocab_size=0, embedding_dim=50, hidden_size=50, n_classes=0, num_layers=2, dropout=0.2):
        super(NER, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers,
                            dropout=dropout)
        self.linear = nn.Linear(hidden_size * 2, n_classes)  # x2 cuz bi-directional

    def forward(self, sentences):
        embedding = self.embedding(sentences)
        lstm, _ = self.lstm(embedding)
        dropout = self.dropout(lstm)
        final_output = self.linear(dropout)
        return final_output
