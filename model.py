import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        #super(DecoderRNN, self).__init__()
        super().__init__()
        
        self.embed_size=embed_size
        self.hidden_size=hidden_size
        self.vocab_size=vocab_size
        self.num_layers=num_layers
        
        self.embed=nn.Embedding(self.vocab_size , self.embed_size)
        self.lstm=nn.LSTM(self.embed_size,self.hidden_size,self.num_layers,batch_first=True)
        self.fc=nn.Linear(self.hidden_size , self.vocab_size)
        self.drop=nn.Dropout(p=0.5)
    
    def forward(self, features, captions):
        batch_size=features.size(0)
        captions=captions[:,:-1]
        caps=self.embed(captions)
        caps = torch.cat((features.unsqueeze(1), caps), dim=1)
                       
        lstm_out,_ =self.lstm(caps)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)
        out=self.drop(lstm_out)
        out=self.fc(out)
        p=out.view(batch_size,-1,self.vocab_size)
        return p 

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        predicted=[]
        lstm_out,h=self.lstm(inputs,states)
        for i in range(max_len):
            
            lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)
            out=self.fc(lstm_out)
            prob=F.softmax(out,dim=1)
            
            p, top_i = prob.topk(k=3)
            top_i = top_i.detach().cpu().numpy().squeeze()
            p = p.detach().cpu().numpy().squeeze()
            word = np.random.choice(top_i, p=p/p.sum())
                       
            word=torch.tensor(word,dtype=torch.long)
            word=word.to(device)
            
            inc=self.embed(word.unsqueeze(0))      
     
            inc=torch.unsqueeze(inc,dim=0)
            lstm_out,h=self.lstm(inc,h)
            predicted.append(int(word))
           
            
        return predicted
        