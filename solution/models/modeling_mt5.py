import torch.nn as nn
from transformers import T5PreTrainedModel
from transformers import MT5EncoderModel
from transformers.modeling_outputs import SequenceClassifierOutput


class MT5ForSequenceClassification(T5PreTrainedModel):
    """ MT5 model for sequence classification
    Attributes:
        num_labels:     The number of labels.
        mt5:            The mt5 model.
        classifier:     The classifier for sequence classification.
        model_parallel: Whether it's parallel or not.
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.mt5 = MT5EncoderModel(config)
        # self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.d_model, config.num_labels)
        self.init_weights()
        self.model_parallel = False
        
    def forward(
        self,
        input_ids=None,
        labels=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.mt5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        last_hidden_state = encoder_outputs[0]
        
        # last_hidden_state = self.dropout(last_hidden_state)
        logits = self.classifier(last_hidden_state[:, -1, :])
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )