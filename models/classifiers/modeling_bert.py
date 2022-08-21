import logging
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple


class ClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    seq_logits: Optional[torch.FloatTensor] = None
    tok_logits: Optional[torch.FloatTensor] = None
    tok_logits1: Optional[torch.FloatTensor] = None
    tok_logits2: Optional[torch.FloatTensor] = None
    seq_loss: Optional[torch.FloatTensor] = None
    tok_loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Pooler(nn.Module):
    def __init__(self, max_seq_length, hidden_size):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        self.dense = nn.Linear(max_seq_length*hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        logging.debug(f'hidden_states: {hidden_states.shape}')
        pooled_output = self.dense(hidden_states.view(-1, self.max_seq_length*self.hidden_size))
        logging.debug(f'pooled_output: {pooled_output.shape}')
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertForUnifiedCR(BertPreTrainedModel):

    def __init__(self, config, max_seq_length=50, num_seq_labels=2, num_tok_labels=5, \
        num_pos_labels=None, num_deprel_labels=None, add_word_and_head_id=False):
        super().__init__(config)
        
        self.num_tok_labels = num_tok_labels
        self.num_seq_labels = num_seq_labels
        self.num_pos_labels = num_pos_labels
        self.num_deprel_labels = num_deprel_labels
        self.add_word_and_head_id = add_word_and_head_id

        self.bert = BertModel(config, add_pooling_layer=False)            
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        total_hidden_size = config.hidden_size
        if num_pos_labels is not None:
            total_hidden_size+=num_pos_labels
        if num_deprel_labels is not None:
            total_hidden_size+=num_deprel_labels
        if add_word_and_head_id:
            total_hidden_size+=2
        lstm_output_size = total_hidden_size//2
        self.reader = nn.LSTM(
            total_hidden_size,
            lstm_output_size,
            bidirectional=True,
            dropout=config.hidden_dropout_prob
        )
        self.linear = nn.Linear(lstm_output_size*2, 256)
        self.linear2 = nn.Linear(256, 64)
        self.tokclf = nn.Linear(64, num_tok_labels)
        self.pool = Pooler(max_seq_length, num_tok_labels)
        self.seqclf = nn.Linear(num_tok_labels, num_seq_labels)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        seq_label=None,
        tok_label=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ignore_index=-100,
        finetune_plm=False,
        teacher_forcing=False,
        tok_class_weights=None,
        pos=None,
        deprel=None,
        word_id=None,
        head_word_id=None,
    ):

        # bert embeddings

        with torch.set_grad_enabled(finetune_plm):
            outputs = self.bert(
                None if input_ids is None else input_ids.view(-1,self.max_seq_length),
                attention_mask=None if attention_mask is None else attention_mask.view(-1,self.max_seq_length),
                token_type_ids=None if token_type_ids is None else token_type_ids.view(-1,self.max_seq_length),
                position_ids=None if position_ids is None else position_ids.view(-1,self.max_seq_length),
                head_mask=None if head_mask is None else head_mask.view(-1,self.max_seq_length),
                inputs_embeds=None if inputs_embeds is None else inputs_embeds.view(-1,self.max_seq_length),
                output_attentions=None if output_attentions is None else output_attentions.view(-1,self.max_seq_length),
                output_hidden_states=None if output_hidden_states is None else output_hidden_states.view(-1,self.max_seq_length),
                return_dict=return_dict
            )
        sequence_output = outputs[0]
        logging.debug(f'sequence_output w/bert: {sequence_output.shape}')  # batch_size, max_seq_length, hidden_size 

        # add other features
        if self.num_pos_labels is not None:
            pos_one_hot = torch.zeros(pos.size(0), pos.size(1), self.num_pos_labels).to(self.device)
            pos_one_hot = pos_one_hot.scatter_(2, pos.unsqueeze(2), 1)
            logging.debug(f'pos_one_hot: {pos_one_hot.shape},  vs self.num_pos_labels: {self.num_pos_labels}')
            sequence_output = torch.cat([sequence_output, pos_one_hot], dim=2)
            logging.debug(f'sequence_output w/pos: {sequence_output.shape}')
            
        if self.num_deprel_labels is not None:
            deprel_one_hot = torch.zeros(deprel.size(0), deprel.size(1), self.num_deprel_labels).to(self.device)
            deprel_one_hot = deprel_one_hot.scatter_(2, deprel.unsqueeze(2), 1)
            logging.debug(f'deprel_one_hot: {deprel_one_hot.shape},  vs self.num_deprel_labels: {self.num_deprel_labels}')
            sequence_output = torch.cat([sequence_output, deprel_one_hot], dim=2)
            logging.debug(f'sequence_output w/deprel: {sequence_output.shape}')
        
        if self.add_word_and_head_id:
            word_id = torch.stack([word_id, head_word_id], dim=2)
            logging.debug(f'word_and_head_id: {word_id.shape}')
            sequence_output = torch.cat([sequence_output, word_id], dim=2)
            logging.debug(f'sequence_output w/deprel: {sequence_output.shape}')
        
        # lstm processing
        sequence_output = sequence_output.permute(1, 0, 2)
        sequence_output, (ht, ct) = self.reader(sequence_output)
        sequence_output = sequence_output.permute(1, 0, 2)
        logging.debug(f'sequence_output w/lstm: {sequence_output.shape}')
        
        # condense
        sequence_output = self.linear(nn.functional.relu(sequence_output))
        sequence_output = self.linear2(nn.functional.relu(sequence_output))
                               
        #####
        hidden_size = sequence_output.shape[2]
        # sequence_output = self.dropout(sequence_output)
        tok_logits = self.tokclf(sequence_output)
        tok_logits = nn.functional.softmax(tok_logits, dim=2)
        
        tok_loss = None
        if tok_label is not None:
            
            tok_loss_fct = nn.CrossEntropyLoss(weight=tok_class_weights, ignore_index=ignore_index)
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = tok_logits.view(-1, self.num_tok_labels)
                active_labels = torch.where(
                    active_loss, tok_label.view(-1), torch.tensor(tok_loss_fct.ignore_index).type_as(tok_label)
                )
                logging.debug(f'active_logits: {active_logits.shape}')
                logging.debug(f'active_labels: {active_labels}')
                tok_loss = tok_loss_fct(active_logits, active_labels)
            else:
                tok_loss = tok_loss_fct(tok_logits.view(-1, self.num_tok_labels), tok_label.view(-1))

        #####
        
        if teacher_forcing and tok_label is not None:
            # replace -100 to 0 ('O')
            # please check that pad_label_idx=-100 and 'O' is indeed represented by 0 in toklabel2id
            pooled_output=self.pool(
                nn.functional.one_hot(
                    torch.where(tok_label>=0, tok_label, 0),
                    num_classes=self.num_tok_labels
                    ).float())
        else:
            pooled_output=self.pool(tok_logits)
        
        logits = self.seqclf(pooled_output)
        logits = nn.functional.softmax(logits, dim=1)

        loss = None
        if seq_label is not None:
            logging.debug(f'logits: {logits.shape}')
            logging.debug(f'seq_label: {seq_label.shape}')
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
            loss = loss_fct(logits.view(-1, self.num_seq_labels), seq_label.view(-1))
        
        #####
        
        return_output = (loss, logits, tok_loss, tok_logits,)

        if not return_dict:
            return_output = (return_output,) + outputs[2:]
        
        return return_output


class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertForUnifiedCRBase(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, num_seq_labels=2, loss_function='simple', alpha=1):
        super().__init__(config)
        self.num_ce_tags = config.num_labels
        self.alpha = alpha

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_ce_tags)
        self.classifier1 = nn.Linear(config.hidden_size+(1*self.num_ce_tags), self.num_ce_tags)
        self.classifier2 = nn.Linear(config.hidden_size+(2*self.num_ce_tags), self.num_ce_tags)

        self.num_seq_labels = num_seq_labels
        self.pooler = BertPooler(config.hidden_size)
        self.seq_classifier = nn.Linear(config.hidden_size, self.num_seq_labels)

        self.function = loss_function
        if self.function=='simple':
            self.loss_fct = nn.CrossEntropyLoss()
        elif self.function=='weighted':
            self.loss_fct = nn.CrossEntropyLoss(reduce=False)
        else:
            raise NotImplementedError('We do not have such a loss function to optimize upon.')
        
        # Initialize weights and apply final processing
        self.post_init()


    def criterion(self, logits, attention_mask, labels):
        # Only keep active parts of the loss
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_ce_tags)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(self.loss_fct.ignore_index).type_as(labels)
            )
            loss = self.loss_fct(active_logits, active_labels)
        else:
            loss = self.loss_fct(logits.view(-1, self.num_ce_tags), labels.view(-1))
        return loss

    
    def torch_mean_ignore_zero(self, y, mask=None):
        if mask is None:
            mask = y!=0
        return (y*mask).sum(dim=-1)/mask.sum(dim=-1)


    def add_if_not_none(self, loss, sub_loss):
        if loss is None:
            return sub_loss
        else:
            return loss+sub_loss


    def calc_loss(self, seq_loss, tok_loss, tok_loss1, tok_loss2, alpha, attention_mask=None):
        
        if self.function == 'simple':
            
            # Simple Loss
            loss = None
            if seq_loss is not None:
                loss = self.add_if_not_none(loss, seq_loss*self.alpha) # increase weight equiv to 3 toks
            if tok_loss is not None:
                loss = self.add_if_not_none(loss, tok_loss)
            if tok_loss1 is not None:
                loss = self.add_if_not_none(loss, tok_loss1)
            if tok_loss2 is not None:
                loss = self.add_if_not_none(loss, tok_loss2)

        elif self.function == 'weighted':

            # OLD CODES: TO REMOVE
            # all examples with no span annotations are already masked
            # Naturally not evaluated against!

            # Weighted Loss
            # To Do: Handle tok_loss1, tok_loss2
            if tok_loss is not None and seq_loss is not None:
                # Assuming no seq examples are masked, we use torch.mean directly
                # Masking only affects token clf (E.g. with pads and no span annotations)
                loss = torch.mean(
                    torch.mul(alpha,self.torch_mean_ignore_zero(
                        tok_loss, attention_mask))
                    + torch.mul(1-alpha,seq_loss)
                    )
            elif tok_loss is not None:
                loss = self.torch_mean_ignore_zero(
                    tok_loss, attention_mask)
            elif seq_loss is not None:
                loss = torch.mean(seq_loss)
            else:
                loss = None
        
        else:

            raise NotImplementedError

        return loss

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        ce_tags=None,
        ce_tags1=None,
        ce_tags2=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        span_egs_mask=None,
        label=None
    ):
        r"""
        ce_tags (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        bs, seq_len = input_ids.shape
        logging.debug(f'input_ids.shape: {input_ids.shape}')
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
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
        sequence_output = self.dropout(sequence_output)
        logging.debug(f'sequence_output.shape: {sequence_output.shape}')
        logits = self.classifier(sequence_output)
        logging.debug(f'logits.shape: {logits.shape}')
        logits1 = self.classifier1(torch.cat((sequence_output,logits),dim=2))
        logging.debug(f'logits1.shape: {logits1.shape}')
        logits2 = self.classifier2(torch.cat((sequence_output,logits,logits1),dim=2))
        logging.debug(f'logits2.shape: {logits2.shape}')
        seq_logits = self.seq_classifier(self.pooler(sequence_output))
        logging.debug(f'seq_logits.shape: {seq_logits.shape}')

        if span_egs_mask is not None:
            # logging.info(f'attention_mask.shape: {attention_mask.shape}')
            # logging.info(f'span_egs_mask.shape: {span_egs_mask.shape}')
            attention_mask = torch.mul(attention_mask, span_egs_mask.unsqueeze(-1))
            alpha = 0.5*span_egs_mask
        else:
            alpha = 0.5

        seq_loss = None
        tok_loss = None
        tok_loss1 = None
        tok_loss2 = None
        # To Do
        if ce_tags is not None:
            tok_loss = self.criterion(logits, attention_mask, ce_tags)
            logging.debug(f'tok_loss value: {tok_loss}')
            if self.function=='weighted':
                tok_loss = tok_loss.view(bs,seq_len)
        if ce_tags1 is not None:
            tok_loss1 = self.criterion(logits1, attention_mask, ce_tags1)
            logging.debug(f'tok_loss1 value: {tok_loss1}')
            if self.function=='weighted':
                tok_loss1 = tok_loss1.view(bs,seq_len) # append by dim=1 at seq_len
        if ce_tags2 is not None:
            tok_loss2 = self.criterion(logits2, attention_mask, ce_tags2)
            logging.debug(f'tok_loss2 value: {tok_loss2}')
            if self.function=='weighted':
                tok_loss2 = tok_loss2.view(bs,seq_len) # append by dim=1 at seq_len
        if label is not None:
            logging.debug(f'label.shape: {label.shape}')
            logging.debug(f'label values: {label}')
            seq_loss = self.loss_fct(seq_logits.view(bs, self.num_seq_labels), label.view(bs))
            logging.debug(f'seq_loss value: {seq_loss}')
        loss = self.calc_loss(seq_loss,
            tok_loss,tok_loss1,tok_loss2,
            alpha,attention_mask
            )
        logging.debug(f'{self.function} loss value: {loss}')

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ClassifierOutput(
            loss=loss,
            seq_loss=seq_loss,
            seq_logits=seq_logits,
            tok_loss=tok_loss,
            tok_logits=logits,
            tok_logits1=logits1,
            tok_logits2=logits2,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertForRevUnifiedCR(BertPreTrainedModel):

    def __init__(self, config, max_seq_length=50, num_seq_labels=2, num_tok_labels=5, \
        num_pos_labels=None, num_deprel_labels=None, add_word_and_head_id=False):
        super().__init__(config)
        
        self.config = config
        self.num_tok_labels = num_tok_labels
        self.num_seq_labels = num_seq_labels
        self.num_pos_labels = num_pos_labels
        self.num_deprel_labels = num_deprel_labels
        self.add_word_and_head_id = add_word_and_head_id
        self.max_seq_length = max_seq_length

        self.bert = BertModel(config, add_pooling_layer=True)            
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # total_hidden_size = config.hidden_size
        # if num_pos_labels is not None:
        #     total_hidden_size+=num_pos_labels
        # if num_deprel_labels is not None:
        #     total_hidden_size+=num_deprel_labels
        # if add_word_and_head_id:
        #     total_hidden_size+=2
        # lstm_output_size = total_hidden_size//2
        # self.reader = nn.LSTM(
        #     total_hidden_size,
        #     lstm_output_size,
        #     bidirectional=True,
        #     dropout=config.hidden_dropout_prob
        # )
        # self.bertpool = BertPooler(config.hidden_size) #lstm_output_size*2)
        # self.linear = nn.Linear(lstm_output_size*2, 256)
        # self.linear2 = nn.Linear(256, 64)
        self.seqclf = nn.Linear(config.hidden_size, num_seq_labels) #64, num_seq_labels)
        self.tokclf = nn.Linear(config.hidden_size, num_tok_labels) #lstm_output_size*2, num_tok_labels)
        # self.pool = Pooler(max_seq_length, num_tok_labels)

        # later versions: use post_init()
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        seq_label=None,
        tok_label=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ignore_index=-100,
        finetune_plm=True,
        teacher_forcing=False,
        tok_class_weights=None,
        pos=None,
        deprel=None,
        word_id=None,
        head_word_id=None,
    ):

        # bert embeddings
        with torch.set_grad_enabled(finetune_plm):

            if self.max_seq_length is None:
                outputs = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict
                )
            else:
                outputs = self.bert(
                    None if input_ids is None else input_ids.view(-1,self.max_seq_length),
                    attention_mask=None if attention_mask is None else attention_mask.view(-1,self.max_seq_length),
                    token_type_ids=None if token_type_ids is None else token_type_ids.view(-1,self.max_seq_length),
                    position_ids=None if position_ids is None else position_ids.view(-1,self.max_seq_length),
                    head_mask=None if head_mask is None else head_mask.view(-1,self.max_seq_length),
                    inputs_embeds=None if inputs_embeds is None else inputs_embeds.view(-1,self.max_seq_length),
                    output_attentions=None if output_attentions is None else output_attentions.view(-1,self.max_seq_length),
                    output_hidden_states=None if output_hidden_states is None else output_hidden_states.view(-1,self.max_seq_length),
                    return_dict=return_dict
                )
        sequence_output = outputs[0]
        logging.debug(f'sequence_output w/bert: {sequence_output.shape}')  # batch_size, max_seq_length, hidden_size 

        # # add other features
        # if self.num_pos_labels is not None:
        #     pos_one_hot = torch.zeros(pos.size(0), pos.size(1), self.num_pos_labels).to(self.device)
        #     pos_one_hot = pos_one_hot.scatter_(2, pos.unsqueeze(2), 1)
        #     logging.debug(f'pos_one_hot: {pos_one_hot.shape},  vs self.num_pos_labels: {self.num_pos_labels}')
        #     sequence_output = torch.cat([sequence_output, pos_one_hot], dim=2)
        #     logging.debug(f'sequence_output w/pos: {sequence_output.shape}')
            
        # if self.num_deprel_labels is not None:
        #     deprel_one_hot = torch.zeros(deprel.size(0), deprel.size(1), self.num_deprel_labels).to(self.device)
        #     deprel_one_hot = deprel_one_hot.scatter_(2, deprel.unsqueeze(2), 1)
        #     logging.debug(f'deprel_one_hot: {deprel_one_hot.shape},  vs self.num_deprel_labels: {self.num_deprel_labels}')
        #     sequence_output = torch.cat([sequence_output, deprel_one_hot], dim=2)
        #     logging.debug(f'sequence_output w/deprel: {sequence_output.shape}')
        
        # if self.add_word_and_head_id:
        #     word_id = torch.stack([word_id, head_word_id], dim=2)
        #     logging.debug(f'word_and_head_id: {word_id.shape}')
        #     sequence_output = torch.cat([sequence_output, word_id], dim=2)
        #     logging.debug(f'sequence_output w/deprel: {sequence_output.shape}')
        
        # # lstm processing
        # sequence_output = sequence_output.permute(1, 0, 2)
        # sequence_output, (ht, ct) = self.reader(sequence_output)
        # sequence_output = sequence_output.permute(1, 0, 2)
        # logging.debug(f'sequence_output w/lstm: {sequence_output.shape}')

        ##### seq clf
                          
        pooled_output = outputs[1]
        # condense
        # pooled_output = self.linear(nn.functional.relu(pooled_output))
        # pooled_output = self.linear2(nn.functional.relu(pooled_output))
        pooled_output = self.dropout(pooled_output)
        logits = self.seqclf(pooled_output)
        # logits = nn.functional.softmax(logits, dim=1)

        total_loss = None
        loss = None
        if seq_label is not None:
            logging.debug(f'logits: {logits.shape}')
            logging.debug(f'seq_label: {seq_label.shape}')
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
            loss = loss_fct(logits.view(-1, self.num_seq_labels), seq_label.view(-1))
            total_loss = loss

        ##### tok clf

        tok_logits = self.tokclf(sequence_output)
        tok_logits = nn.functional.softmax(tok_logits, dim=2)
        
        tok_loss = None
        # if tok_label is not None:
            
        #     tok_loss_fct = nn.CrossEntropyLoss(weight=tok_class_weights, ignore_index=ignore_index)
        #     if attention_mask is not None:
        #         active_loss = attention_mask.view(-1) == 1
        #         active_logits = tok_logits.view(-1, self.num_tok_labels)
        #         active_labels = torch.where(
        #             active_loss, tok_label.view(-1), torch.tensor(tok_loss_fct.ignore_index).type_as(tok_label)
        #         )
        #         logging.debug(f'active_logits: {active_logits.shape}')
        #         logging.debug(f'active_labels: {active_labels}')
        #         tok_loss = tok_loss_fct(active_logits, active_labels)
        #     else:
        #         tok_loss = tok_loss_fct(tok_logits.view(-1, self.num_tok_labels), tok_label.view(-1))
            
        #     if total_loss is None:
        #         total_loss = tok_loss
        #     else:
        #         total_loss += tok_loss
        
        #####
    
        return_output = (loss, logits, tok_loss, tok_logits,)

        if not return_dict:
            return_output = (return_output,) + outputs[2:]
        
        return ClassifierOutput(
            loss=total_loss,
            seq_loss=loss,
            seq_logits=logits,
            tok_loss=tok_loss,
            tok_logits=tok_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )