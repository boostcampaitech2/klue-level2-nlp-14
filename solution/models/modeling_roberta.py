import math
from collections import OrderedDict, UserDict

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_utils import PreTrainedModel
from transformers.models.roberta.modeling_roberta import (
  RobertaModel,
  RobertaPreTrainedModel,
  RobertaForSequenceClassification,
)
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import (
  BaseModelOutputWithPoolingAndCrossAttentions,
  SequenceClassifierOutput,
)


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
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
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


class RobertaForSequenceClassificationLstm(RobertaPreTrainedModel):
    """ Roberta model for sequence classification with LSTM classifier head """
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lstm = nn.LSTM(input_size = 1024, hidden_size = 1024, num_layers = 3, dropout=0.5, bidirectional = True, batch_first = True)
        self.dense_layer = nn.Linear(2048, 30, bias=True)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
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
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
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

        enc_hiddens, (last_hidden, last_cell) = self.lstm(sequence_output)
        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim = 1)

        logits = self.dense_layer(output_hidden)

#         logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
