class biLSTM(nn.Module):
    def __init__(self,vocab_size, embedding_dim, hidden_size, num_classes, num_layers,bidirectional):
        super(biLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(n_class, embedding_dim=emb_size)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size,batch_first=True,
                            num_layers=self.num_layers,bidirectional=self.bidirectional)
        if self.bidirectional:
            self.fc = nn.Linear(hidden_size*2, num_classes)
        else:
            self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        batch_size, seq_len ,b= x.shape
        #初始化一个h0,也即c0，在RNN中一个Cell输出的ht和Ct是相同的，而LSTM的一个cell输出的ht和Ct是不同的
        #维度[layers, batch, hidden_len]
        if self.bidirectional:
            h0 = torch.randn(self.num_layers*2, batch_size, self.hidden_size).to(device)
            c0 = torch.randn(self.num_layers*2, batch_size, self.hidden_size).to(device)
        else:
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
            c0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
        x = self.embedding(x)
        x = x.transpose(0, 1)
        out,(_,_)= self.lstm(x, (h0,c0))
        output = self.fc(out[:,-1,:]).squeeze(0) #因为有max_seq_len个时态，所以取最后一个时态即-1层
        return output
