import torch
import torch.nn as nn

class Classifier(nn.Module):

    def __init__(self, num_labels, args):
        super(Classifier, self).__init__()

        self.dropout = args.dropout
        self.gru_hidden_dim = args.gru_hidden_dim 
        
        if args.gru_bidirectional:
            self.gru_hidden_dim = 2 * args.gru_hidden_dim
        
        self.linear = nn.Linear(self.gru_hidden_dim, self.gru_hidden_dim)
        self.bn = nn.BatchNorm1d(self.gru_hidden_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()
        self.output = nn.Linear(self.gru_hidden_dim, num_labels)

    def forward(self, model_output):
        x = self.linear(model_output)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


class CRNN_Model(nn.Module):

    def __init__(self, num_labels, vocab_size, args):
        super(CRNN_Model, self).__init__()

        self.num_labels = num_labels
        self.vocab_size = vocab_size
    
        self.embed_dim = args.embed_dim          # in_channels : 300
        self.num_kernels = args.num_kernels      # out_channels : 100
        self.kernel_sizes = args.kernel_sizes    # kernel_sizes [3,4,5]
        self.stride = args.stride                # stride: 1
        self.gru_hidden_dim = args.gru_hidden_dim
        self.gru_bidirectional = args.gru_bidirectional

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.embed_dim,
                                      padding_idx = 0)
        
        self.cnn1d_layers = nn.ModuleList([
            self.make_cnn1d_sequential(in_channels=self.embed_dim,
                                       out_channels=self.num_kernels,
                                       kernel_size=ks,
                                       stride=self.stride) for ks in self.kernel_sizes
        ])
    
        self.gru = nn.GRU(input_size=self.num_kernels,
                          hidden_size=self.gru_hidden_dim,
                          bidirectional=self.gru_bidirectional,
                          batch_first=True,
        )

        self.classifier = Classifier(self.num_labels, args)
    
    @staticmethod
    def make_cnn1d_sequential(in_channels, out_channels, kernel_size, stride):

        padding = (kernel_size - 1) // 2
        
        return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=padding),
                             nn.BatchNorm1d(out_channels),
                             nn.ReLU(),
                             nn.MaxPool1d(kernel_size, stride)
                            )

    def forward(self, input_ids):
        x = self.embedding(input_ids)   # (batch_size, seq_length, embed_dim)
        x = x.permute(0, 2, 1)          # (batch_size, embed_dim, seq_length)

        conv_outs = [cnn_layer(x) for cnn_layer in self.cnn1d_layers] # conv_out : (batch_size, out_channels, conv_out_length)
        
        x = torch.cat(conv_outs, dim=2) # x : (batch_size, out_channels, total_conv_out_length)
        x = x.permute(0, 2, 1)          # x : (batch_size, total_conv_out_length, out_channels)

        self.gru.flatten_parameters()
        
        gru_out, _ = self.gru(x)  # gru_out : (batch_size, total_conv_out_length, gru_hidden_dim)
        x = gru_out[:,-1,:]       # x : (batch_size, gru_hidden_dim)
         
        x = self.classifier(x)
        
        return x      