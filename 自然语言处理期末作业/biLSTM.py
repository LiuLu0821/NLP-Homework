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
        #��ʼ��һ��h0,Ҳ��c0����RNN��һ��Cell�����ht��Ct����ͬ�ģ���LSTM��һ��cell�����ht��Ct�ǲ�ͬ��
        #ά��[layers, batch, hidden_len]
        if self.bidirectional:
            h0 = torch.randn(self.num_layers*2, batch_size, self.hidden_size).to(device)
            c0 = torch.randn(self.num_layers*2, batch_size, self.hidden_size).to(device)
        else:
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
            c0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
        x = self.embedding(x)
        x = x.transpose(0, 1)
        out,(_,_)= self.lstm(x, (h0,c0))
        output = self.fc(out[:,-1,:]).squeeze(0) #��Ϊ��max_seq_len��ʱ̬������ȡ���һ��ʱ̬��-1��
        return output