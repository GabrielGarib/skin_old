import torch
import torch.nn as nn

class MetaEffnet(nn.Module):

    def __init__(self, effnet, num_classes, concat=False, n_metadata=0, neurons_reducer_block=256,
                 p_dropout=0.5, n_feat_conv=1280):

        super().__init__()

        self.concat = concat
        self.effnet = effnet
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

        # Feature reducer
        if neurons_reducer_block > 0:
            self.reducer_block = nn.Sequential(
                nn.Linear(n_feat_conv, neurons_reducer_block),
                nn.BatchNorm1d(neurons_reducer_block),
                nn.ReLU(),
                nn.Dropout(p=p_dropout)
            )
            self.classifier = nn.Linear(neurons_reducer_block + n_metadata, num_classes)
        else:
            self.reducer_block = None
            self.classifier = nn.Linear(n_feat_conv + n_metadata, num_classes)

    def forward(self, img, metadata=None):
        x = self.effnet.extract_features(img)
        x = self.avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatting

        if self.concat:
            if self.reducer_block is not None:
                x = self.reducer_block(x)
            x = torch.cat([x, metadata], dim=1) # concatenation
        else:
            if self.reducer_block is not None:
                x = self.reducer_block(x)  # feat reducer block
        x = x.float()
        return self.classifier(x)
