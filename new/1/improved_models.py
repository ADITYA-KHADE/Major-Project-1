import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, RobertaModel, RobertaTokenizer

class BERTHateSpeechClassifier(nn.Module):
    def __init__(self, pretrained_model="distilbert-base-uncased", num_classes=2):
        super(BERTHateSpeechClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class RoBERTaHateSpeechClassifier(nn.Module):
    def __init__(self, pretrained_model="roberta-base", num_classes=2):
        super(RoBERTaHateSpeechClassifier, self).__init__()
        self.encoder = RobertaModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class EnsembleModel(nn.Module):
    def __init__(self, lstm_cnn_model, bert_model, bert_tokenizer):
        super(EnsembleModel, self).__init__()
        self.lstm_cnn = lstm_cnn_model
        self.bert = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.classifier = nn.Linear(4, 2)  # Combines outputs from both models
        
    def forward(self, text, sequence):
        # Get LSTM+CNN prediction
        lstm_output = self.lstm_cnn(sequence)
        
        # Get BERT prediction
        tokenized = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        bert_output = self.bert(tokenized['input_ids'], tokenized['attention_mask'])
        
        # Combine predictions
        combined = torch.cat([lstm_output, bert_output], dim=1)
        return self.classifier(combined)

class EnhancedEnsembleModel(nn.Module):
    def __init__(self, lstm_cnn_model, roberta_model, roberta_tokenizer):
        super(EnhancedEnsembleModel, self).__init__()
        self.lstm_cnn = lstm_cnn_model
        self.roberta = roberta_model
        self.roberta_tokenizer = roberta_tokenizer
        self.classifier = nn.Linear(4, 2)  # Combines outputs from both models
        
    def forward(self, text, sequence):
        # Get LSTM+CNN prediction
        lstm_output = self.lstm_cnn(sequence)
        
        # Get RoBERTa prediction
        tokenized = self.roberta_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        roberta_output = self.roberta(tokenized['input_ids'], tokenized['attention_mask'])
        
        # Combine predictions
        combined = torch.cat([lstm_output, roberta_output], dim=1)
        return self.classifier(combined)
