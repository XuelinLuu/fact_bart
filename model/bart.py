import math
from typing import Optional, Tuple
import torch
from torch import nn
from torch import random
from torch._C import device
from torch.nn.modules import dropout, padding


def shift_tokens_right(input_ids: torch.Tensor, pad_token_ids: int, decoder_start_token_id: int):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 1] = decoder_start_token_id

    assert pad_token_ids is not None, 'pad_token_id has to be defined'
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_ids)
    return shifted_input_ids

def _expand_mask(mask: torch.Tensor, dtype=torch.dtype, tgt_len: Optional[int]=None):
    batch_size, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expand_mask = mask[:, None, None, :].expand(batch_size, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expand_mask
    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


def make_causal_mask(input_shape_size: torch.Size, dtype: torch.dtype, past_key_values_length: int= 0):
    # past_key_values_length: attn -> src_len
    batch_size, tgt_len = input_shape_size
    mask = torch.full((tgt_len, tgt_len), float('-inf')) # mask 全部初始化为 -inf
    mask_cond = torch.arnage(mask.size(-1)) # size = (tgt_len,)
    # 对角矩阵，主对角线以及下三角全部为 0 
    # mask_cond -> (tgt_len, )
    # mask_cond.view(tgt_len, 1)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        # 此处past_key_value_length 全部置为 0 是因为前面内容对后面均可见
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    # mask size: [tgt_len, (past_key_value_length + tgt_len)]
    return mask[None, None, :, :].expand(batch_size, 1, tgt_len, tgt_len + past_key_values_length)

class BartModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        padding_idx, vocab_size = args.pad_token_id, args.vocab_size
        self.shared = nn.Embedding(vocab_size, args.d_model, padding_idx)

        self.encoder = BartEncoder(args, self.shared)
        self.decoder = BartDecoder(args, self.shared)

        self.init_weights()

    def forward(
        self, 
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,  # TODO
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids=input_ids,
                pad_token_ids=self.args.pad_token_id,
                decoder_start_token_id=self.args.decoder_start_token_id
            )
        output_attentions = output_attentions if output_attentions is not None else self.args.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.args.output_hidden_states

        use_cache = use_cache if use_cache is not None else self.args.use_cache

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask = attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states
            )

        decoder_outputs = self.decoder(
            input_ids = decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        return encoder_outputs + decoder_outputs

    def init_weights():
        pass


class BartEncoder(nn.Module):
    def __init__(self, args, embed_tokens):
        super().__init__()

        self.args = args
        self.dropout = args.dropout
        self.layerdrop = args.encoder_layerdrop

        embed_dim = args.d_model
        self.padding_idx = args.pad_token_id
        self.max_source_position = args.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if args.scale_embedding else 1.0

        self.emebd_tokens = embed_tokens  # xl
        self.embed_position = BartLearnedPositionalEmbedding(args.max_position_embeddings, embed_dim)  # [1026, embedding_dim]
        
        self.layers = nn.ModuleList([BartEncoderLayer(args) for _ in range(args.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        # self.init_weights()

    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.args.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.args.output_hidden_states

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('Can not specify both input_ids and input_embeds at the same time')
        elif input_ids is not None:
            input_shape = input_ids.size()  # [batch_size, sequence_len]
            input_ids = input_ids.view(-1, input_shape[-1]) # [-1, sequence_len]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:2]
        else:
            raise ValueError('Have to specify either input_ids or input_embeds')
            
        if inputs_embeds is None:  # [batch_size, sequence_len, hidden_size]
            inputs_embeds = self.emebd_tokens(input_ids) * self.embed_scale
        
        embed_pos = self.embed_position(input_shape) # [batch_size, sequence_len, hidden_size]

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if attention_mask is not None:
            # [batch_size, sequence_len] -> [batch_size, 1, trg_len, src_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if head_mask is not None:
            assert head_mask.size()[0] == len(self.layers), 'head_mask size[0] != num_layers'

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states += (hidden_states, )

            layer_outputs = encoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=head_mask,
                output_attentions=output_attentions
            )
            hidden_states = layer_outputs[0]
        
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1])
        
        if output_hidden_states:
            encoder_states += (hidden_states)
        
        return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)

class BartDecoder(nn.Module):
    def __init__(self, args, embed_tokens):
        super().__init__()
        self.args = args()
        self.dropout = args.dropout
        self.layerdrop = args.decoder_layerdrop
        self.padding_idx = args.pad_token_id
        self.max_target_positions = args.max_position_embeddings
        self.embed_scale = math.sqrt(args.d_model) if args.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(args.vocab_size, args.d_model, self.padding_idx)
        
        self.embed_positions = BartLearnedPositionalEmbedding(
            args.max_position_embeddings,
            args.d_model
        )

        self.layers = nn.ModuleList([BartDecoderLayer(args) for _ in range(args.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(args.d_model)

        self.init_weights()
    
    def init_weights(self):
        pass
    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_value_length):
        # create causal mask
        # [batch_size, seq_len] -> [batch_size, 1, seq_len, src_len]
        # src_len = seq_len + past_key_value_length

        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = make_causal_mask(input_shape, inputs_embeds.dtype, past_key_value_length).to(device)
        
        if attention_mask is not None:
            expand_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, input_shape[-1])
            combined_attention_mask = (
                expand_attn_mask if combined_attention_mask is None else combined_attention_mask + expand_attn_mask
            )
        
        return combined_attention_mask

        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False
    ):
        output_attentions = output_attentions if output_attentions is not None else self.args.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.args.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.args.use_cache
        return_dict = return_dict if return_dict is not None else self.args.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('Can not specify both decoder_input_ids and decoder_inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:2]
        else:
            raise ValueError('Have to specify either decoder_input_ids or decoder_inputs_embeds')

        # TODO analysis past key value
        # past_key_value = (key_state, value_state, cross_key_states, cross_value_states)
        # (batch_size, num_heads, tgt_len, src_len)
        # past_key_values_length = src_len
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        
        # 此处添加了了 causal mask 和 attention mask
        attention_mask = self._prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length)

        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [batch_size, sequence_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # TODO: tgt position embeds 为什么要加上 past key value 的长度src_len
        # position最大允许1024，emcoder 的 length 超过1024再加上岂不是越界了？？？
        positions = self.embed_positions(input_shape, past_key_values_length)
        
        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder_layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (
                    len(self.layers)
                ), f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_values = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=(head_mask[idx if head_mask is not None else None]),
                cross_attn_layer_head_mask=(
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                ),
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1], )
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if encoder_hidden_states is not None:
                all_cross_attentions += (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
            if v is not None
        )

class BartEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.embed_dim = args.d_model
        self.self_attn = BartAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout  
        )

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = args.dropout
        self.activate_fn = nn.GELU()
        self.activation_dropout = args.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, args.encoder_ffn_dim)
        self.fc2 = nn.Linear(args.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)


    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch_size, sequence_len, embed_dim]
        attention_mask: torch.Tensor,  # [batch_size, 1, sequence_len, sequence_len]
        layer_head_mask: torch.Tensor,  # mask for attention heads in a given layer of size, (encoder_attention_heads_num,)
        output_attentions: bool=False
    ):
        # @return:
        #   hidden_states
        #   attention weight (optional)

        # self-attn
        residual = hidden_states
        # self-attn: 
        #   hidden_states([batch_size, sequence_len, hidden_dim]), 
        #   attn_weight([batch_size, sequence_len, sequence_len])
        hidden_states, attn_weight = self.self_attn(
            hidden_states=hidden_states,
            key_value_states=None,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions
        ) 
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states += residual
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # feedforward fc
        residual = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activate_fn(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states += residual
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weight,)
        return outputs


class BartDecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            is_decoder=True
        )

        self.dropout = args.dropout
        self.activation_fn = nn.GELU()
        self.activation_dropout = args.activation_dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.encoder_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            is_decoder=True
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, args.decoder_ffn_dim)
        self.fc2 = nn.Linear(self.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor]=None,
        encoder_hidden_states: Optional[torch.Tensor]=None,
        encoder_attention_mask: Optional[torch.Tensor]=None,
        layer_head_mask: Optional[torch.Tensor]=None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):
        # self-attention
        residual = hidden_states
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # cross-attention
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )

            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            present_key_value = present_key_value + cross_attn_present_key_value

        residual = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states, )

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class BartAttention(nn.Module):
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        dropout: float=0.0, 
        is_decoder: bool=False, 
        bias: bool=True
        ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (self.head_dim * num_heads == embed_dim), 'Attention num_head * head_dim != embed_dim'

        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linead(embed_dim, embed_dim, bias=bias)
    
    def reshape_proj(self, proj_states):
        # [batch_size, sequence_length, hidden_size] ->
        # [batch_size, sequence_length, num_head, head_dim] ->
        # [batch_size, num_head, sequence_length, head_dim]
        batch_size, sequence_len = proj_states.size()[:2]
        new_shape = [batch_size, sequence_len, self.num_heads, self.head_dim]
        proj_states = proj_states.view(*new_shape)
        return proj_states.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor]=None,
        past_key_value: Optional[Tuple[torch.Tensor]]=None, # TODO
        attention_mask: Optional[torch.Tensor]=None,
        layer_head_mask: Optional[torch.Tensor]=None,
        output_attentions: bool=False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        is_cross_attention = key_value_states is not None  # key_value_states 表示encoder的输入，如果存在，则表示未decoder-attention
        batch_size, target_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:  
            # cross attention and encoder inputs(k & v) is not None
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross attention
            key_states = self.k_proj(key_value_states)
            value_states = self.v_proj(key_value_states)
        elif past_key_value:
            key_states = self.k_proj(hidden_states)
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = self.v_proj(hidden_states)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)
        else:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        query_states = self.reshape_proj(query_states)
        key_states = self.reshape_proj(key_states)
        value_states = self.reshape_proj(value_states)

        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))

        if attention_mask is not None:
            attn_weights += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights

        if output_attentions:
            attn_weights_output = attn_weights
        else:
            attn_weights_output = None


        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        
        attn_output = torch.matmul(attn_probs, value_states) # [batch_size, num_head, sequence_len, head_dim]
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.view(batch_size, target_len, embed_dim)

        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights_output


class BartLearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        # embedding dim = [1026, embedding_dim]
        # 1026 = offset + max_position_embeddings 
        self.offset = 2
        super().__init__(num_embeddings+self.offset, embedding_dim)
    
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int=0):
        # [batch size, sequence length]
        # past_key_values_length TODO
        batch_size, sequence_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + sequence_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions + self.offset)

