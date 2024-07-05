from transformers.models.bert.modeling_bert import *


class BertBinaryClassifier(BertPreTrainedModel):
    def __init__(self, transformer_name, config, num_labels):
        super().__init__(config)

        self.model = BertModel.from_pretrained(transformer_name)
        self.num_labels = num_labels

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )

        self.dropout = nn.Dropout(classifier_dropout)
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.ac1 = nn.GELU()
        self.ac2 = nn.GELU()

        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = True, ):

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs.last_hidden_state
        batch_size = pooled_output.shape[0]
        pooled_output = pooled_output[torch.arange(batch_size, device=pooled_output.device), 0]

        pooled_output = self.dropout(pooled_output)
        pooled_output = self.fc1(pooled_output)
        pooled_output = self.ac1(pooled_output)
        pooled_output = self.fc2(pooled_output)
        pooled_output = self.ac2(pooled_output)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
