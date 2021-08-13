import torch
import math
from typing import Optional, Tuple
from torch import nn


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class FactBartForGeneration(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = FactBartModel(args)
        self.lm_head = nn.Linear(args.d_model, self.model.shared.num_embeddings, bias=False)

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def forward(
            self,
            article_input_ids=None,  # input_ids
            article_attention_mask=None,  # attention_mask
            highlights_input_ids=None,  # decoder_input_ids
            highlights_attention_mask=None,  # decoder_attention_mask
            article_triples_input_ids=None,
            article_triples_attention_mask=None,
            article_triples_start_positions=None,
            article_triples_start_positions_mask=None,
            article_triples_label=None,
            article_triples_label_mask=None,

            article_head_mask=None,
            highlights_head_mask=None,
            article_triples_head_mask=None,

            past_key_values=None,

            article_input_embeds=None,
            highlights_input_embeds=None,
            article_triples_input_embeds=None,

            use_cache=None,
            output_attentions=None,
            output_hidden_states=None

    ):
        outputs = self.model(
            article_input_ids=article_input_ids,  # input_ids
            article_attention_mask=article_attention_mask,  # attention_mask
            highlights_input_ids=highlights_input_ids,  # decoder_input_ids
            highlights_attention_mask=highlights_attention_mask,  # decoder_attention_mask
            article_triples_input_ids=article_triples_input_ids,
            article_triples_attention_mask=article_triples_attention_mask,
            article_triples_start_positions=article_triples_start_positions,
            article_triples_start_positions_mask=article_triples_start_positions_mask,

            article_head_mask=article_head_mask,
            highlights_head_mask=highlights_head_mask,
            article_triples_head_mask=article_triples_head_mask,

            past_key_values=past_key_values,

            article_input_embeds=article_input_embeds,
            highlights_input_embeds=highlights_input_embeds,
            article_triples_input_embeds=article_triples_input_embeds,

            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        # [batch_size, sequence_len, hidden_size]
        decoder_output = outputs['decoder_output']
        lm_logits = self.lm_head(decoder_output[0])
        fact_logits = outputs['triples_start_logits']

        loss_ = None
        if highlights_input_ids is not None or article_triples_label is not None:
            if highlights_input_ids is not None:
                loss_lm = nn.CrossEntropyLoss()
                masked_lm_loss = loss_lm(lm_logits.view(-1, self.args.vocab_size), highlights_input_ids.view(-1))
            else:
                masked_lm_loss = 0

            if article_triples_label is not None:
                loss_fc = nn.CrossEntropyLoss()
                masked_fc_loss = loss_fc(fact_logits.view(-1, 2), article_triples_label.view(-1))
            else:
                masked_fc_loss = 0

            loss_ = masked_lm_loss + masked_fc_loss

        output = (lm_logits, ) + decoder_output
        return (loss_, ) + output if loss_ is not None else output


class FactBartModel(nn.Module):
    def __init__(self, args):
        super(FactBartModel, self).__init__()
        self.args = args
        # padding_idx = 1 <pad> = 1
        padding_idx, vocab_size = args.pad_token_id, args.vocab_size
        # word embedding
        self.shared = nn.Embedding(vocab_size, args.d_model, padding_idx)

        self.encoder = FactBartEncoder(args, self.shared)
        self.decoder = FactBartDecoder(args, self.shared)

        self.triples_classifier = nn.Linear(self.args.d_model, 2, bias=False)

        self.init_weights()

    def init_weights(self):
        # TODO
        pass

    def get_input_embeddings(self):
        return self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def shift_tokens_right(self, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        # input_ids: [batch_size, seq_len]
        shifted_input_ids = torch.zeros_like(input_ids)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        assert pad_token_id is not None, 'pad_token_id has to be defined'
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def forward(
            self,
            article_input_ids=None,  # input_ids
            article_attention_mask=None,  # attention_mask
            highlights_input_ids=None,  # decoder_input_ids
            highlights_attention_mask=None,  # decoder_attention_mask
            article_triples_input_ids=None,
            article_triples_attention_mask=None,
            article_triples_start_positions=None,
            article_triples_start_positions_mask=None,

            article_head_mask=None,
            highlights_head_mask=None,
            article_triples_head_mask=None,

            past_key_values=None,

            article_input_embeds=None,
            highlights_input_embeds=None,
            article_triples_input_embeds=None,

            use_cache=None,
            output_attentions=None,
            output_hidden_states=None
    ):
        if highlights_input_ids is not None and highlights_input_embeds is None:
            # shifted right
            highlights_input_ids = self.shift_tokens_right(
                input_ids=highlights_input_ids,
                pad_token_id=self.args.pad_token_id,
                decoder_start_token_id=self.args.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.args.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.args.output_hidden_states

        use_cache = use_cache if use_cache is not None else self.args.use_cache

        # encoder hidden states: [batch_size, seq_len, hidden_size]
        article_encoder_output = self.encoder(
            input_ids=article_input_ids,
            attention_mask=article_attention_mask,
            head_mask=article_head_mask,
            input_embeds=article_input_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        triples_encoder_output = self.encoder(
            input_ids=article_triples_input_ids,
            attention_mask=article_triples_attention_mask,
            head_mask=article_triples_head_mask,
            input_embeds=article_triples_input_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        # triple_start_positions triple_start_mask
        # 通过triple_mask获取
        triples_hidden_states = triples_encoder_output[0]
        batch_size, seq_len, hidden_size = triples_hidden_states.size()[:3]
        # triple_start_tokens: [batch_size, num_triples_per_document, hidden_states]
        triples_start_tokens = triples_hidden_states[torch.arange(batch_size).unsqueeze(1), article_triples_start_positions]
        # 三元组的 start_hidden_states: [batch_size, num_triples_per_document, hidden_states]
        # 所有三元组，无论是不是 label = 1
        triples_start_tokens = triples_start_tokens * article_triples_start_positions_mask.unsqueeze(-1)  # => triples_hidden_states
        # triples_start_logits: [batch_size, num_triples_per_document, 2]
        triples_start_logits = self.triples_classifier(triples_start_tokens)
        triples_start_logits = nn.Softmax(dim=-1)(triples_start_logits)
        # triples_start_selected_probs: [batch_size, num_triples_per_document, 1]
        triples_start_logits = triples_start_logits * article_triples_start_positions_mask.unsqueeze(-1)
        triples_start_selected_probs = torch.select(triples_start_logits, dim=-1, index=1)

        triples_start_selected = torch.where(
            triples_start_selected_probs > 0.5,
            torch.ones_like(triples_start_selected_probs),
            torch.zeros_like(triples_start_selected_probs)
        )

        with torch.no_grad():
            triples_start_selected = triples_start_selected - triples_start_selected_probs
        # triples_start_selected => triples_attention_mask
        triples_start_selected = triples_start_selected + triples_start_selected_probs

        decoder_output = self.decoder(
            input_ids=highlights_input_ids,
            attention_mask=highlights_attention_mask,
            head_mask=highlights_head_mask,

            article_hidden_states=article_encoder_output[0],
            article_attention_mask=article_attention_mask,  # [batch_size, sequence_len]
            article_head_mask=article_head_mask,

            triples_hidden_states=triples_start_tokens,
            triples_attention_mask=triples_start_selected,
            triples_head_mask=article_triples_head_mask,

            past_key_values=past_key_values,
            input_embeds=highlights_input_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        return {
            # 'last_hidden_states': decoder_output[0],
            # 'past_key_values': decoder_output[1],
            'decoder_output': decoder_output,  # (hidden_states, past_key_values, all_hidden_states, all_attention_weights, )
            'article_encoder_output': article_encoder_output,
            'triples_encoder_output': triples_encoder_output,
            'triples_start_logits': triples_start_logits
        }


class FactBartEncoder(nn.Module):
    def __init__(self, args, embed_tokens):
        super().__init__()
        self.args = args
        self.dropout = args.dropout
        self.layerdrop = args.encoder_layerdrop

        embed_dim = args.d_model
        self.padding_idx = args.pad_token_id  # <pad> = 1
        self.max_source_position = args.max_position_embeddings  # max_position = 1024

        self.embed_tokens = embed_tokens
        self.embed_position = FactBartLearningPositionEmbedding(self.max_source_position, embed_dim)
        # self.embed_triple = FactBartLearningTripleEmbedding(embed_dim)
        self.layers = nn.ModuleList([FactBartEncoderLayer(args) for _ in range(args.num_encoder_layers)])
        self.embedding_layernorm = nn.LayerNorm(embed_dim)

    def _expand_mask(self, attn_mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        # self-attn: tgt_len = src_len, cross_attn: tgt != src_len
        batch_size, src_len = attn_mask.size()  # [batch_size, src_len]
        tgt_len = tgt_len if tgt_len is not None else src_len
        # [batch_size, src_len] -> [batch_size, 1, tgt_len, src_len]
        expanded_mask = attn_mask[:, None, None, :].expand(batch_size, 1, tgt_len, src_len).to(dtype)
        inverted_mask = 1.0 - expanded_mask
        final_mask = inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)
        return final_mask

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            input_embeds=None,
            output_attentions=None,
            output_hidden_states=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.args.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.args.output_hidden_states
        if input_ids is not None and input_embeds is not None:
            raise ValueError('Can not specify both input_ids and input_embeds at the same time')
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif input_embeds is not None:
            input_shape = input_embeds.size()[:2]
        else:
            raise ValueError('Have to specify either input_ids or input_embeds')

        if input_embeds is None:  # word embedding
            input_embeds = self.embed_tokens(input_ids)

        position_embeds = self.embed_position(input_shape)

        hidden_states = input_embeds + position_embeds
        hidden_states = self.embedding_layernorm(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if attention_mask is not None:
            # attn mask: [batch_size, sequence_len] -> [batch_size, 1, tgt_len, src_len]
            attention_mask = self._expand_mask(attention_mask, input_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if head_mask is not None:
            assert head_mask.size()[0] == len(self.layers), 'head_mask size[0] != num layers'

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states += (hidden_states, )

            layer_output = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                output_attentions=output_attentions
            )
            hidden_states = layer_output[0]
            if output_attentions:
                all_attentions += (layer_output[1])
        if output_hidden_states:
            encoder_states += (hidden_states, )
        return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)


class FactBartDecoder(nn.Module):
    def __init__(self, args, embed_tokens=None):
        super().__init__()
        self.args = args
        self.dropout = args.dropout
        self.layerdrop = args.decoder_layerdrop
        self.padding_idx = args.pad_token_id
        self.max_tgt_positions = args.max_position_embeddings
        self.embed_scale = math.sqrt(args.d_model)

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(args.vocab_size, args.d_model, self.padding_idx)
        self.embed_positions = FactBartLearningPositionEmbedding(self.max_tgt_positions, args.d_model)

        self.layers = nn.ModuleList([FactBartDecoderLayer(args) for _ in range(args.num_decoder_layers)])
        self.embedding_layernorm = nn.LayerNorm(args.d_model)

    def _make_casual_mask(self, input_shape: torch.Size, dtype: torch.dtype, past_key_value_length: int):
        # decoder 对角mask
        batch_size, tgt_len = input_shape[:2]
        mask = torch.full((tgt_len, tgt_len), float('-inf'))  # 初始化全部mask为 -∞
        mask_cond = torch.arange(mask.size()[-1])  # [0, ..., tgt_len-1]
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size()[-1], 1), 0)
        mask = mask.to(dtype)
        if past_key_value_length != 0:
            # [tgt_len, past_len] + [tgt_len, tgt_len] = [tgt_len, past_len + tgt_len]
            mask = torch.cat([torch.zeros(size=(tgt_len, past_key_value_length), dtype=dtype), mask], dim=-1)
        return mask[None, None, :, :].expand(batch_size, 1, tgt_len, tgt_len + past_key_value_length)

    def _expand_mask(self, attn_mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        # self-attn: tgt_len = src_len, cross_attn: tgt != src_len
        batch_size, src_len = attn_mask.size()  # [batch_size, src_len]
        tgt_len = tgt_len if tgt_len is not None else src_len
        # [batch_size, src_len] -> [batch_size, 1, tgt_len, src_len]
        expanded_mask = attn_mask[:, None, None, :].expand(batch_size, 1, tgt_len, src_len).to(dtype)
        inverted_mask = 1.0 - expanded_mask
        final_mask = inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)
        return final_mask

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, input_embeds, past_key_value_length):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            # make_casual_mask: self_mask 对角矩阵 返回 非mask 为0， mask 为 float("-inf") size: [batch_size, 1, tgt_len, tgt_len + past_key_value_len]
            combined_attention_mask = self._make_casual_mask(
                input_shape,
                input_embeds.dtype,
                past_key_value_length
            )

        if attention_mask is not None:
            # attention_mask: [batch_size, arc_len] -> [batch_size, 1, tgt_len, src_len]
            # params: [batch_size, src_len] dtype (tgt_len, )
            expand_attention_mask = self._expand_mask(attention_mask, input_embeds.dtype, input_shape[-1])
            # past_key_value_len = 0，所以可以进行相加
            # TODO: 分析能否删掉past_key_value_len
            combined_attention_mask = (
                expand_attention_mask if combined_attention_mask is None else expand_attention_mask + combined_attention_mask
            )
        return combined_attention_mask

    def forward(
            self,
            input_ids: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            head_mask:torch.Tensor = None,

            article_hidden_states: torch.Tensor = None,
            article_attention_mask: torch.Tensor = None,
            article_head_mask: torch.Tensor = None,

            triples_hidden_states: torch.Tensor = None,
            triples_attention_mask: torch.Tensor = None,
            triples_head_mask: torch.Tensor = None,

            past_key_values=None,
            input_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.args.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.args.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.args.use_cache

        if input_ids is not None and input_embeds is not None:
            raise ValueError('Can not specify both decoder_input_ids and decoder_inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif input_embeds is not None:
            input_shape = input_embeds.size()[:2]
        else:
            raise ValueError('Have to specify either decoder_input_ids or decoder_inputs_embeds')

        # past_key_values: (key_states, value_states, fact_key_states, fact_value_states, cross_key_states, cross_value_states)
        # -> [batch_size, num_head, tgt_len, src_len] * 2
        # [0] -> [batch_size, num_head, tgt_len, src_len]
        # [0] -> [num_head, tgt_len, src_len]
        # shape[2] -> src_len
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids) * self.embed_scale
        # attention_mask [batch_size, src_len] -> batch_size, sequence_len
        # TODO: 分析BartModel decoder_input_ids 和 decoder_attention_mask
        attention_mask = self._prepare_decoder_attention_mask(attention_mask, input_shape, input_embeds, past_key_values_length)

        if article_hidden_states is not None and article_attention_mask is not None:
            # [batch_size, src_len] -> [batch_size, 1, tgt_len, src_len]
            article_attention_mask = self._expand_mask(article_attention_mask, input_embeds.dtype, tgt_len=input_shape[-1])

        if triples_hidden_states is not None and triples_attention_mask is not None:
            # [batch_size, src_len] -> [batch_size, 1, tgt_len, src_len]
            triples_attention_mask = self._expand_mask(triples_attention_mask, input_embeds.dtype, tgt_len=input_shape[-1])

        positions = self.embed_positions(input_shape)

        hidden_states = input_embeds + positions
        hidden_states = self.embedding_layernorm(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layer
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states)
            # TODO: 20210809** 分析 past_key_values 如何传入 bart_model 然后返回值做何种处理
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,  # [batch_size, sequence_len, hidden_size]
                attention_mask=attention_mask,  # [batch_size, 1, tgt_len, src_len]
                article_hidden_states=article_hidden_states,  # [batch_size, sequence_len, hidden_size]
                article_attention_mask=article_attention_mask,  # [batch_size, 1, tgt_len, src_len]
                triples_hidden_states=triples_hidden_states,  # [batch_size, sequence_len, hidden_size]
                triples_attention_mask=triples_attention_mask,  # [batch_size, 1, tgt_len, src_len]
                layer_head_mask=head_mask[idx] if head_mask is not None else None,
                layer_article_head_mask=article_head_mask[idx] if article_head_mask is not None else None,
                layer_triples_head_mask=triples_head_mask[idx] if triples_head_mask is not None else None,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1], )

            if output_attentions:
                all_attentions += (layer_outputs[1: -1])
        if output_hidden_states:
            all_hidden_states += (hidden_states, )
        # TODO: 20210812 next_cache 是 num_layers * present_value 的一维数组
        next_cache = next_decoder_cache if use_cache else None

        final_outputs = (hidden_states, next_cache)
        if output_hidden_states:
            final_outputs += (all_hidden_states, )
        if output_attentions:
            final_outputs += (all_attentions, )

        return final_outputs


class FactBartDecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.d_model

        self.self_attn = FactBartMutilHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_num_attention_heads,
            dropout=args.attention_dropout,
            is_decoder=True
        )

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.triple_attn = FactBartMutilHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_num_attention_heads,
            dropout=args.attention_dropout,
            is_decoder=True
        )
        self.triple_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.article_attn = FactBartMutilHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_num_attention_heads,
            dropout=args.attention_dropout,
            is_decoder=True
        )
        self.article_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.dropout = args.dropout
        self.fc1 = nn.Linear(self.embed_dim, args.decoder_ffn_dim)
        self.activation_fn = gelu
        self.activation_dropout = args.activation_dropout
        self.fc2 = nn.Linear(args.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,  # [batch_size, sequence_len, hidden_size]
            attention_mask: Optional[torch.Tensor] = None,  # [batch_size, 1, tgt_len, src_len]
            article_hidden_states: Optional[torch.Tensor] = None,  # [batch_size, sequence_len, hidden_size]
            article_attention_mask: Optional[torch.Tensor] = None,  # [batch_size, 1, tgt_len, src_len]
            triples_hidden_states: Optional[torch.Tensor] = None,  # [batch_size, sequence_len, hidden_size]
            triples_attention_mask: Optional[torch.Tensor] = None,  # [batch_size, 1, tgt_len, src_len]
            layer_head_mask: Optional[torch.Tensor] = None,
            layer_article_head_mask: Optional[torch.Tensor] = None,
            layer_triples_head_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = True,
    ):
        residual = hidden_states
        # Self Attention
        # decoder uni-directional self-attention cached key/value tuple is at position 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        hidden_states, self_attn_weight, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = hidden_states + residual
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Triples Attention

        residual = hidden_states
        triples_attn_past_key_value = past_key_value[2: 4] if past_key_value is not None else None
        hidden_states, triples_attn_weights, triples_attn_present_key_value = self.triple_attn(
            hidden_states=hidden_states,
            key_value_states=triples_hidden_states,
            attention_mask=triples_attention_mask,
            layer_head_mask=layer_triples_head_mask,
            past_key_value=triples_attn_past_key_value,
            output_attentions=output_attentions
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.triple_attn_layer_norm(hidden_states)

        # article Attention
        residual = hidden_states
        article_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
        hidden_states, article_attn_weights, article_attn_present_Key_value = self.article_attn(
            hidden_states=hidden_states,
            key_value_states=article_hidden_states,
            attention_mask=article_attention_mask,
            layer_head_mask=layer_article_head_mask,
            past_key_value=article_attn_past_key_value,
            output_attentions=output_attentions
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.article_attn_layer_norm(hidden_states)

        present_key_value = present_key_value + triples_attn_present_key_value + article_attn_present_Key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weight, triples_attn_weights, article_attn_weights)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


class FactBartEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.d_model
        self.self_attn = FactBartMutilHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.encoder_num_attention_heads,
            dropout=args.attention_dropout
        )
        self.self_attn_layernorm = nn.LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, args.encoder_ffn_dim)
        self.activate_fn = gelu
        self.dropout = args.dropout
        self.fc2 = nn.Linear(args.encoder_ffn_dim, self.embed_dim)
        self.activation_dropout = args.activation_dropout
        self.final_layernorm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            layer_head_mask: torch.Tensor,
            output_attentions: bool=False
    ):
        # @return
        # hidden_states
        # attention_mask(optional)

        # self-attention
        residual = hidden_states

        hidden_states, attn_weight = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states += residual
        hidden_states = self.self_attn_layernorm(hidden_states)

        # feed-forward
        residual = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activate_fn(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = hidden_states + residual
        hidden_states = self.final_layernorm(hidden_states)

        outputs = (hidden_states, attn_weight) if output_attentions else (hidden_states, )

        return outputs


class FactBartMutilHeadAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.is_decoder = is_decoder
        assert (self.embed_dim == self.num_heads * self.head_dim), 'num_heads * head_dim should be equal to embed_dim'

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)

    def reshape(self, proj_states):
        # proj_states: [batch_size, sequence_len, embed_dim]
        batch_size, sequence_len, embed_dim = proj_states.size()
        new_shape = [batch_size, sequence_len, self.num_heads, self.head_dim]
        proj_states = proj_states.view(*new_shape)
        return proj_states.permute(0, 2, 1, 3)  # [batch_size, num_head, seq_len, head_dim]

    def back_shape(self, proj_states):
        # proj_states: [batch_size, num_head, seq_len, head_dim]
        batch_size, num_head, seq_len, head_dim = proj_states.size()
        proj_states = proj_states.permute(0, 2, 1, 3)  # [batch_size, seq_len, num_heads, head_dim]
        old_shape = [batch_size, seq_len, self.embed_dim]
        proj_states = proj_states.contiguous().view(*old_shape)
        return proj_states

    def forward(
            self,
            hidden_states,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False
    ):
        # past_key_value 不为空时，表示 decoder 上一轮的结果存储，
        # decoder-self-attn 存储上一轮 self-attn key/value
        # decoder-article-attn 存储上一轮 article-attn key/value
        # decoder-triples-attn 存储上一轮 triples-attn key/value
        query_states = self.q_proj(hidden_states)  # [batch_size, tgt_len, hidden_dim]
        query_states = self.reshape(query_states)  # [batch_size, num_heads, tgt_len, head_dim]
        # if cross-attention, the keys and values is from encoder, the attention mask needs to be such that
        # the encoder's padding tokens are not attended to
        is_cross_attention = bool(key_value_states is not None)

        # TODO: 要改动一下，cross 是 document 还是 triples
        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]  # [batch_size, num_heads, src_len, head_dim]
            value_states = past_key_value[1]  # [batch_size, num_heads, src_len, head_dim]
        elif is_cross_attention:
            # is_cross_attention and past_key_value is None:
            key_states = self.k_proj(key_value_states)  # [batch_size, src_len, hidden_dim]
            value_states = self.v_proj(key_value_states)  # [batch_size, src_len, hidden_dim]
            key_states = self.reshape(key_states)  # [batch_size, num_heads, src_len, head_dim]
            value_states = self.reshape(value_states)  # [batch_size, num_heads, src_len, head_dim]
        elif past_key_value is not None:
            # self-attention and past_key_value is not None => decoder self-attention
            key_states = self.k_proj(hidden_states)  # [batch_size, src_len, hidden_dim]
            value_states = self.v_proj(hidden_states)  # [batch_size, src_len, hidden_dim]
            key_states = self.reshape(key_states)  # [batch_size, num_heads, src_len, head_dim]
            value_states = self.reshape(value_states)  # [batch_size, num_heads, src_len, head_dim]

            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self-attention and past_key_value is None => decoder self-attention
            key_states = self.k_proj(hidden_states)  # [batch_size, src_len, hidden_dim]
            value_states = self.v_proj(hidden_states)  # [batch_size, src_len, hidden_dim]
            key_states = self.reshape(key_states)  # [batch_size, num_heads, src_len, head_dim]
            value_states = self.reshape(value_states)  # [batch_size, num_heads, src_len, head_dim]

        if self.is_decoder:
            # if cross_attention save Tuple[torch.Tensor, torch.Tensor] of all cross attention key/value_states
            # Further calls to cross_attention layer can then reuse all cross_attention key/value_states (first 'if' case)
            # if uni-directional self-attention(decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention can concat
            # previous decoder key/value_states to current projected key/value_states(third 'elif' case)
            # if encoder bi-directional self-attention 'past_key_value' is always None
            past_key_value = (key_states, value_states)

        # [batch_size, num_heads, tgt_len, head_dim] * [batch_size, num_heads, head_dim, src_len]
        # => [batch_size, num_heads, tgt_len, src_len]
        attention_score = torch.matmul(query_states, key_states.transpose(-1, -2))

        # attention_mask: [batch_size, 1, tgt_len, src_len]
        if attention_mask is not None:
            attention_score += attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_score)
        attention_probs = nn.Dropout(p=self.dropout)(attention_probs)

        if layer_head_mask is not None:
            attention_probs = attention_probs * layer_head_mask

        attn_output = torch.matmul(attention_probs, value_states)
        attn_output = self.back_shape(attn_output)

        attn_output = self.out_proj(attn_output)

        outputs = (attn_output, attention_probs) if output_attentions else (attn_output, None)

        if self.is_decoder:
            outputs += (past_key_value, )
        return outputs


class FactBartLearningPositionEmbedding(nn.Embedding):
    def __init__(self, max_position: int, embed_dim: int):
        super().__init__(num_embeddings=max_position, embedding_dim=embed_dim)

    def forward(self, input_ids_shape: torch.Size):
        batch_size, sequence_len = input_ids_shape[:2]
        positions = torch.arange(sequence_len, dtype=torch.int64, device=self.weight.device)
        return super().forward(positions)


class FactBartLearningTripleEmbedding(nn.Embedding):
    def __init__(self, embed_dim: int):
        super().__init__(num_embeddings=2, embedding_dim=embed_dim)

    def forward(self, triple_ids):
        return super().forward(triple_ids)