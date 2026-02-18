"""
BERT Model Wrapper
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizer
from typing import Optional, Dict


class BERTWrapper(nn.Module):
    """
    Wrapper for BERT model with configurable settings for profiling.
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased', 
                 num_labels: int = 2, pretrained: bool = True):
        """
        Initialize BERT model.
        
        Args:
            model_name: Name of BERT model ('bert-base-uncased', 'bert-large-uncased', etc.)
            num_labels: Number of classification labels
            pretrained: Whether to use pretrained weights
        """
        super(BERTWrapper, self).__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load BERT model
        if pretrained:
            self.bert = BertModel.from_pretrained(model_name)
        else:
            config = BertConfig.from_pretrained(model_name)
            self.bert = BertModel(config)
        
        # Classification head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through BERT.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            token_type_ids: Token type IDs [batch_size, seq_length]
            
        Returns:
            Logits [batch_size, num_labels]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def get_model_info(self) -> Dict:
        """Return information about the model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        param_memory = sum(p.numel() * p.element_size() for p in self.parameters())
        
        return {
            'name': self.model_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'param_memory_mb': param_memory / (1024 ** 2),
            'num_labels': self.num_labels,
            'hidden_size': self.bert.config.hidden_size,
            'num_layers': self.bert.config.num_hidden_layers,
            'num_attention_heads': self.bert.config.num_attention_heads
        }
    
    def print_model_info(self):
        """Print model information."""
        info = self.get_model_info()
        print(f"\n{'='*60}")
        print(f"Model: {info['name']}")
        print(f"{'='*60}")
        print(f"Total Parameters: {info['total_params']:,}")
        print(f"Trainable Parameters: {info['trainable_params']:,}")
        print(f"Parameter Memory: {info['param_memory_mb']:.2f} MB")
        print(f"Number of Labels: {info['num_labels']}")
        print(f"Hidden Size: {info['hidden_size']}")
        print(f"Number of Layers: {info['num_layers']}")
        print(f"Attention Heads: {info['num_attention_heads']}")
        print(f"{'='*60}\n")


def create_bert(model_name: str = 'bert-base-uncased', num_labels: int = 2,
                pretrained: bool = True, device: str = 'cpu') -> BERTWrapper:
    """
    Factory function to create BERT model.
    
    Args:
        model_name: BERT model variant
        num_labels: Number of classification labels
        pretrained: Whether to use pretrained weights
        device: Device to place model on ('cpu', 'cuda', 'mps')
        
    Returns:
        BERTWrapper instance
    """
    model = BERTWrapper(model_name=model_name, num_labels=num_labels, pretrained=pretrained)
    model = model.to(device)
    return model


def get_bert_tokenizer(model_name: str = 'bert-base-uncased'):
    """
    Get tokenizer for BERT model.
    
    Args:
        model_name: BERT model variant
        
    Returns:
        BertTokenizer instance
    """
    return BertTokenizer.from_pretrained(model_name)
