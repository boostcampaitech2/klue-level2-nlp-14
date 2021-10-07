import torch
import torch.nn as nn
from transformers import (
    RobertaPreTrainedModel,
    RobertaModel,
    RobertaForSequenceClassification,
)
from transformers.modeling_outputs import SequenceClassifierOutput


class ClassificationLinearHead(nn.Module):
    """ Head for sentence-level Linear classification head """
    
    def __init__(
        self, 
        hid_dim: int, 
        num_labels: int,
        act_type: str,
        classifier_dropout: float,
    ):
        super().__init__()
        self.activation = getattr(torch, act_type)
        self.dense = nn.Linear(hid_dim, hid_dim * 4)
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(hid_dim * 4, num_labels)
        
    def forward(self, features, **kwargs):
        x = features[:, 0, :] # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
            
            
class ClassificationLSTMHead(nn.Module):
    """ Head for sentence-level LSTM classification head """
    
    def __init__(
        self, 
        hid_dim: int, 
        num_labels: int,
        act_type: str,
        classifier_dropout: float,
    ):
        super().__init__()
        self.activation = getattr(torch, act_type)
        self.dense = nn.LSTM(hid_dim, hid_dim//2,
                             batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(hid_dim, num_labels)
        
    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x, _ = self.dense(x)
        x = x[:, -1, :] # take last hidden state
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
        
class RecentMultiClassificationHead(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.dense_type in ["Linear", "LSTM"]
        assert config.act_type in ["tanh", "relu"]
        act_type = config.act_type
        hid_dim = config.hidden_size
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        if config.dense_type == "Linear":
            DenseClass = ClassificationLinearHead
        else:
            DenseClass = ClassificationLSTMHead
        heads = []
        for num_labels in config.num_labels_per_head:
            heads.append(DenseClass(hid_dim, num_labels, act_type, classifier_dropout))
        self.heads = nn.ModuleList(heads)
        
    def forward(self, features, **kwargs):
        logits = {}
        for i, head in enumerate(self.heads):
            logits[i] = head(features)
        return logits


class RobertaForKlueRecent(RobertaPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.roberta = RobertaModel(config, add_pooling_layer=True)
        self.classifier = RecentMultiClassificationHead(config)
        
        self.init_weights()
        
    def forward(
        self,
        input_ids=None,
        head_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        head_logits = self.classifier(sequence_output)
        
        # first, scatter
        scattered = []
        bsz = input_ids.shape[0]
        for logit, ind in zip(head_logits.values(), self.config.head2label):
            z = torch.ones(bsz, self.config.num_labels,
                           device=logit.device, dtype=logit.dtype) * 1e-07
            ind = torch.tensor(ind, device=input_ids.device).repeat(bsz, 1)
            scattered.append(z.scatter(1, ind, logit))
        del z, ind, logit
            
        # second gather
        cat_logits = torch.cat(
            [tensor.view(bsz, 1, -1) for tensor in scattered],
            dim=1,
        )
        del scattered
        ind = head_ids.detach().view(-1, 1, 1)
        ind = ind.repeat(1, 1, self.config.num_labels)
        logits = cat_logits.gather(-2, ind).squeeze()
        del ind, cat_logits
        torch.cuda.empty_cache()
        
        has_labels = labels is not None and labels.shape[-1] != 1
        loss = None
        if has_labels:
            for ix, n_labels in enumerate(self.config.num_labels_per_head):
                weight = torch.tensor([0.1] + [1] * (n_labels - 1),
                                      device=logits.device)
                loss_fct = nn.CrossEntropyLoss(weight=weight)
                label = labels[:, ix+1]
                head_loss = loss_fct(head_logits[ix], label)
                if loss is None:
                    loss = head_loss
                else:
                    loss += head_loss
                    
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
                    
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )