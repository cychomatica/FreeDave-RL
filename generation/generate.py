import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import GenerationMixin
from transformers.cache_utils import Cache
from transformers.utils import logging
from typing import Optional, List, Union
from deprecated import deprecated
from utils import sample_tokens
import math
from transformers.cache_utils import DynamicCache

logger = logging.get_logger(__name__)


def generate_FreeDave(model: PreTrainedModel, **kwargs):
    pass


class DLM_Generator:

    '''
        Notes: now only consider 1 token per step
    '''

    def __init__(self, model: PreTrainedModel, generation_config: GenerationConfig):
        self.model = model
        self.generation_config = generation_config
        self.device = model.device
        self.token_per_step = 1

    def get_processed_model_outputs(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.LongTensor] = None, 
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor], Cache] = None,
        use_cache: bool = False,
        store_kv: bool = False,
        output_attentions: bool = False,
        dual_cache: bool = False,
        replace_position: Optional[torch.Tensor] = None
    ):

        '''
            For AR-adapted DLM, we need to process the model logits due to the logits shift.
        '''

        if self.model.__class__.__name__ in ['DreamModel']:
            model_output = self.model(
                x, 
                attention_mask, 
                position_ids, 
                past_key_values, 
                use_cache=use_cache, 
                output_attentions=output_attentions,
                dual_cache=dual_cache, 
                replace_position=replace_position
            )
            logits = model_output.logits
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
            past_key_values = model_output.past_key_values if use_cache else None
            attentions = model_output.attentions if output_attentions else None

            return logits, past_key_values, attentions

        else:
            model_output = self.model(
                x, 
                attention_mask, 
                position_ids, 
                past_key_values, 
                use_cache=use_cache, 
                output_attentions=output_attentions, 
                store_kv=store_kv
            )
            logits = model_output.logits
            past_key_values = model_output.past_key_values if use_cache else None
            attentions = model_output.attentions if output_attentions else None

            return logits, past_key_values, attentions

    def get_attention_mask_and_position_ids(
        self, 
        attention_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        block_length: Optional[int] = None, 
        num_blocks: Optional[int] = None, 
        attention_type: Optional[str] = None
    ):
        
        if attention_type == 'full':
            if attention_mask is not None and torch.any(attention_mask == 0.0): # when input_ids is padded
                attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                attention_mask = torch.logical_and(
                    attention_mask.unsqueeze(1).unsqueeze(-2),
                    attention_mask.unsqueeze(1).unsqueeze(-1),
                )
            else:
                attention_mask = None
                position_ids = None
        elif attention_type == 'block_causal':
            if block_length is None or num_blocks is None:
                raise ValueError('block_length and num_blocks must be provided when attention_type is block_causal')
            inter_block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=self.device))
            attention_mask = inter_block_mask.repeat_interleave(block_length, dim=0).repeat_interleave(block_length, dim=1).unsqueeze(0).unsqueeze(0) # (1, 1, num_blocks * block_length, num_blocks * block_length)
            position_ids = torch.arange(max_length, device=self.device).unsqueeze(0)
        else:
            logger.warning(f'Attention type {attention_type} not supported, using full attention (attention_mask = None) by default.')
            attention_mask = None
            position_ids = None
        return attention_mask, position_ids

    def create_tree_attention_mask_and_position_ids(
        self, 
        attention_mask: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.Tensor] = None, 
        shared_prefix_length: Optional[int] = None,
        branch_length: Optional[int] = None,
        shared_suffix_length: Optional[int] = None,
        num_branches: Optional[int] = None
    ):
        '''
        Create tree-structured attention mask and position ids for a block tree.
        TODO: support full attention DLM
        '''
        if attention_mask.ndim == 4:
            attn_mask = attention_mask.squeeze(0).squeeze(0)
        assert attn_mask.ndim == 2, 'attention_mask must be 2D'
        assert attn_mask.shape == (branch_length, shared_prefix_length + branch_length + shared_suffix_length), 'attention_mask must be of shape (branch_length, shared_prefix_length + branch_length + shared_suffix_length)'
        
        tree_attention_mask = torch.ones(branch_length * num_branches, shared_prefix_length + branch_length * num_branches + shared_suffix_length)
        tree_attention_mask[:, shared_prefix_length:tree_attention_mask.shape[1]-shared_suffix_length] = torch.block_diag(*[attn_mask[:, shared_prefix_length:tree_attention_mask.shape[1]-shared_suffix_length]] * num_branches) 

        tree_position_ids = position_ids.repeat(1, num_branches) # (1, branch_length * num_branches)

        return tree_attention_mask, tree_position_ids

    # TODO: deprecate this function, now only consider 1 token per step
    @deprecated('This function is deprecated, now only consider 1 token per step')
    def get_num_transfer_tokens(block_length, steps):
        base = block_length // steps
        remainder = block_length % steps
        num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
        num_transfer_tokens[:remainder] += 1
        return num_transfer_tokens

    def get_mask_token_prediction_and_confidence(
        self,
        logits: torch.Tensor,
        mask_index: torch.Tensor,
        mask_token_id: int,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ):

        confidence = torch.full_like(mask_index, -torch.inf, device=self.device, dtype=logits.dtype)
        x0 = torch.full_like(mask_index, mask_token_id)
        mask_logits = logits[mask_index]
        mask_confidence, mask_x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
        confidence[mask_index] = mask_confidence
        x0[mask_index] = mask_x0
        return x0, confidence

    def token_transfer(
        self, 
        x: torch.Tensor, 
        x0: torch.Tensor, 
        confidence: torch.Tensor, 
        block_length: int,
        mask_token_id: int, 
        confidence_priority_shift: Optional[torch.Tensor] = None,
        alg_temp: Optional[float] = None,
        confidence_threshold: Optional[float] = None, 
        token_per_step: Optional[int] = 1,
        draft_steps: Optional[int] = None,
    ):
        '''
        Returns the updated x after token transfer.
        Args:
            x: torch.Tensor, the original sequence
            x0: torch.Tensor, the sampled token
            confidence: torch.Tensor, the confidence scores of the sampled tokens
            mask_token_id: int, the mask token id
            alg_temp: Optional[float], the temperature for token transferring
            confidence_threshold: Optional[float], the confidence threshold for token transferring
            num_transfer_tokens: Union[torch.Tensor, List[int]], a 1-D tensor representing the numbers of transfer tokens. If len(num_transfer_tokens) > 1, multiple draft sequences will be decoded.
        '''

        mask_index = (x == mask_token_id)

        if confidence_threshold is not None:
            # take all tokens with confidence >= confidence_threshold
            # TODO: FreeDave v2
            pass
    
        else:
            if draft_steps is not None and draft_steps > 1:
                x_draft = x.unsqueeze(1).expand(x.shape[0], draft_steps, *x.shape[1:]).clone() # (batch, draft_steps, seq_len)
            else:
                x_draft = x.unsqueeze(1).clone() # (batch, 1, seq_len)
            transfer_index = torch.zeros_like(x_draft, dtype=torch.bool)

            if confidence_priority_shift is not None:
                confidence = confidence + confidence_priority_shift

            if alg_temp is None or alg_temp == 0: # maskgit: unmask tokens with highest confidence
                for j in range(confidence.shape[0]):
                    for k in range(draft_steps):
                        _, idx = torch.topk(confidence[j], token_per_step * (k+1))
                        transfer_index[j, k, idx] = True
            elif alg_temp == -1: # l2r: unmask left to right
                for j in range(confidence.shape[0]):
                    for k in range(draft_steps):
                        idx = torch.arange(x.size(1), device=self.device)[mask_index[0]][:token_per_step * (k+1)]
                        transfer_index[j, k, idx] = True
            else:
                raise NotImplementedError(f'Algorithm temperature {alg_temp} not supported yet. ')

            if draft_steps is not None and draft_steps > 1:
                x_draft[transfer_index] = x0.unsqueeze(1).expand_as(x_draft)[transfer_index] # (batch, draft_steps, seq_len)
                x_draft = x_draft.view(x.shape[0] * draft_steps, *x.shape[1:]) # (batch * draft_steps, seq_len)
            else:
                x_draft[transfer_index] = x0.unsqueeze(1)[transfer_index] # (batch, 1, seq_len)
                x_draft = x_draft.squeeze(1) # (batch, seq_len)

            return x_draft


    @torch.no_grad()
    def cache_batch_repeat_interleave(past_key_values, n):
        '''
            past_key_values: 
            a list of n_heads tuples (key, value)
            key: (batch, seq_len, head_dim)
            value: (batch, seq_len, head_dim)

            new_past_key_values:
            a list of n_heads tuples (key, value)
            key: (batch * n, seq_len, head_dim)
            value: (batch * n, seq_len, head_dim)
        '''
        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j].expand(n, *past_key_values[i][j].shape[1:]),)
        return new_past_key_values

    @torch.no_grad()
    def cache_batch_select_indices(past_key_values, indices):
        '''
            past_key_values: 
            a list of n_heads tuples (key, value)
            key: (batch, seq_len, head_dim)
            value: (batch, seq_len, head_dim)

            new_past_key_values:
            a list of n_heads tuples (key, value)
            key: (batch, seq_len, head_dim)
            value: (batch, seq_len, head_dim)
        '''
        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][indices, ...],)
        return new_past_key_values

    @torch.no_grad()
    def block_decode_with_full_attention(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.LongTensor],
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        alg_temp: Optional[float] = None,
        block_length: Optional[int] = 32,
        max_gen_length: Optional[int] = 128,
        decoding_steps: Optional[int] = 128,
        use_cache: bool = False,
        mask_token_id: int = 151666,
        eos_token_id: int = 151645,
        pad_token_id: int = 151643,
        pad_target_penalty: float = 1.0,
        confidence_threshold: Optional[float] = None,
        dual_cache: bool = False
    ):
        '''
        NOTE: now only consider 1 token per step; all the token transfer operations are performed in the block region. 
        '''

        prompt_length = input_ids.shape[1]
        max_length = prompt_length + max_gen_length
        x = F.pad(input_ids, (0, max_gen_length), value=mask_token_id)
        
        if block_length is None:
            block_length = max_gen_length

        assert max_gen_length % block_length == 0, 'max_gen_length {} must be divisible by block_length {}'.format(max_gen_length, block_length)
        num_gen_blocks = max_gen_length // block_length

        assert decoding_steps % num_gen_blocks == 0, 'decoding_steps {} must be divisible by num_gen_blocks {}'.format(decoding_steps, num_gen_blocks)
        steps_per_block = decoding_steps // num_gen_blocks
        # assert steps_per_block == block_length, 'steps_per_block must be equal to block_length for now'
        # num_transfer_tokens_per_block = self.get_num_transfer_tokens(block_length, steps_per_block)

        attention_mask, position_ids = self.get_attention_mask_and_position_ids(
            attention_mask=attention_mask,
            max_length=max_length,
            block_length=block_length,
            num_blocks=num_gen_blocks,
            attention_type='full'
        )
        past_key_values = None

        for block_idx in range(num_gen_blocks):

            current_block_start = prompt_length + block_idx * block_length
            current_block_end = current_block_start + block_length
            current_block_x = x[:, current_block_start:current_block_end].clone()

            # update cache and decode the first step in the current block
            logits, past_key_values, attentions = self.get_processed_model_outputs(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=False
            )
            confidence, x0 = sample_tokens(logits, temperature=1.0, top_k=None, top_p=None)
            x[:, current_block_start] = x0[:, current_block_start] # we must always unmask the first token in the current block due to the logits shift of AR-adapted DLM
                        
            if not dual_cache: # Extract only previous block cache
                new_past_key_values = []
                for i in range(len(past_key_values)):
                    new_past_key_values.append(())
                    for j in range(len(past_key_values[i])):
                        new_past_key_values[i] += (past_key_values[i][j][:, :current_block_start, :],)
                past_key_values = new_past_key_values
            else:
                replace_position = torch.zeros_like(x, dtype=torch.bool)
                replace_position[:, current_block_start:current_block_end] = 1

            # decoding loop for the current block
            step = 1
            while step < steps_per_block:

                current_block_mask_index = (current_block_x == mask_token_id)
                
                # Prepare attention mask for cached generation
                if attention_mask != "full":
                    # Adjust attention mask for current position
                    current_attention_mask = attention_mask[:, :, :, current_block_start:]
                else:
                    current_attention_mask = attention_mask
                
                if dual_cache:
                    # only use the current block for forward pass
                    logits, past_key_values, attentions = self.get_processed_model_outputs(
                        x[:, current_block_start:current_block_end], 
                        current_attention_mask, 
                        position_ids[:, current_block_start:current_block_end] if position_ids is not None else None, 
                        past_key_values=past_key_values, 
                        use_cache=True, 
                        dual_cache=dual_cache, 
                        replace_position=replace_position
                    )
                else:
                    # use the entire remaining sequence for forward pass
                    logits, past_key_values, attentions = self.get_processed_model_outputs(
                        x[:, current_block_start:], 
                        current_attention_mask, 
                        position_ids[:, current_block_start:] if position_ids is not None else None, 
                        past_key_values=past_key_values, 
                        use_cache=True
                    )

                current_block_x0, current_block_confidence = self.get_mask_token_prediction_and_confidence(
                    logits=logits[..., :block_length],
                    mask_index=current_block_mask_index,
                    mask_token_id=mask_token_id,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
                # current_block_confidence = torch.full_like(current_block_x, -torch.inf, device=self.device, dtype=logits.dtype)
                # current_block_x0 = torch.full_like(current_block_x, mask_token_id)
                # current_block_mask_logits = logits[..., :block_length][current_block_mask_index]
                # current_block_confidence_masked, current_block_x0_masked = sample_tokens(current_block_mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                # current_block_confidence[current_block_mask_index] = current_block_confidence_masked
                # current_block_x0[current_block_mask_index] = current_block_x0_masked.clone()

                current_block_x = self.token_transfer(
                    x=current_block_x,
                    x0=current_block_x0,
                    confidence=current_block_confidence,
                    block_length=block_length,
                    mask_token_id=mask_token_id,
                    alg_temp=alg_temp,
                    token_per_step=self.token_per_step,
                )

                if (current_block_x == mask_token_id).all():
                    break

            # commit the current block to the sequence
            x[:, current_block_start:current_block_end] = current_block_x
        
        return x

    @torch.no_grad()
    def block_decode_with_block_causal_attention(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.LongTensor],
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        alg_temp: Optional[float] = None,
        block_length: Optional[int] = 32,
        max_gen_length: Optional[int] = 128,
        decoding_steps: Optional[int] = 128,
        use_cache: bool = False,
        mask_token_id: int = 151666,
        eos_token_id: int = 151645,
        pad_token_id: int = 151643,
        pad_target_penalty: float = 1.0,
        confidence_threshold: Optional[float] = None,
    ):
        '''
        block_decode_with_block_causal_attention for block-causal models like SDAR and TraDo
        NOTE: now only consider 1 token per step; all the token transfer operations are performed in the block region. 
        '''

        batch_size = input_ids.shape[0]
        assert batch_size == 1, 'batch size must be 1 for block_decode_with_block_causal_attention'

        prompt_length = input_ids.shape[1]
        num_total_blocks = (prompt_length + max_gen_length + block_length - 1) // block_length # pad the whole sequence to be divisible by block_length
        max_length = num_total_blocks * block_length
        x = F.pad(input_ids, (0, max_length - prompt_length), value=mask_token_id)

        num_prefill_blocks = prompt_length // block_length
        prefill_length = num_prefill_blocks * block_length
        
        assert decoding_steps % num_total_blocks == 0, 'decoding_steps {} must be divisible by num_total_blocks {}'.format(decoding_steps, num_total_blocks)
        steps_per_block = decoding_steps // num_total_blocks

        attention_mask, position_ids = self.get_attention_mask_and_position_ids(
            attention_mask=attention_mask,
            max_length=max_length,
            block_length=block_length,
            num_blocks=num_total_blocks,
            attention_type='block_causal'
        )
        past_key_values = None

        # Prefilling stage
        if prefill_length > 0:
            cur_x = x[:, :prefill_length]
            cur_attn_mask = attention_mask[..., prefill_length, :prefill_length]
            if cur_attn_mask.dim() == 3:
                cur_attn_mask = cur_attn_mask[:, None, :, :]
            cur_position_ids = position_ids[:, :prefill_length]
            logits, past_key_values, attentions = self.get_processed_model_outputs(
                cur_x,
                attention_mask=cur_attn_mask,
                position_ids=cur_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                store_kv=True if use_cache else False
            )

        # Decoding stage
        for block_idx in range(num_prefill_blocks, num_total_blocks):

            current_block_start = block_idx * block_length
            current_block_end = current_block_start + block_length
            current_block_attention_mask = attention_mask[..., current_block_start:current_block_end, :current_block_end]
            if current_block_attention_mask.dim() == 3:
                current_block_attention_mask = current_block_attention_mask[:, None, :, :]
            current_block_position_ids = position_ids[:, current_block_start:current_block_end]
            current_block_x = x[:, current_block_start:current_block_end].clone()
            
            # decoding loop for the current block
            for step in range(steps_per_block):

                current_block_mask_index = (current_block_x == mask_token_id)

                logits, _, _ = self.get_processed_model_outputs(
                    current_block_x,
                    attention_mask=current_block_attention_mask,
                    position_ids=current_block_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=False
                )

                current_block_x0, current_block_confidence = self.get_mask_token_prediction_and_confidence(
                    logits=logits,
                    mask_index=current_block_mask_index,
                    mask_token_id=mask_token_id,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )

                current_block_x = self.token_transfer(
                    x=current_block_x,
                    x0=current_block_x0,
                    confidence=current_block_confidence,
                    block_length=block_length,
                    mask_token_id=mask_token_id,
                    alg_temp=alg_temp,
                    token_per_step=self.token_per_step,
                )

                if (current_block_x == mask_token_id).all():
                    # if the current block is all unmasked, store the current block's kv and exit the decoding loop
                    _, past_key_values, _ = self.get_processed_model_outputs(
                        current_block_x,
                        current_block_attention_mask,
                        current_block_position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        store_kv=True
                    )
                    break

            # commit the current block to the sequence
            x[:, current_block_start:current_block_end] = current_block_x
        
        return x

    @torch.no_grad()
    def block_decode_with_block_causal_attention_FreeDave(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.LongTensor],
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        alg_temp: Optional[float] = None,
        block_length: Optional[int] = 32,
        max_gen_length: Optional[int] = 128,
        decoding_steps: Optional[int] = 128,
        draft_steps: Optional[int] = 1,
        eager_acceptance_mode: Optional[bool] = False,
        use_cache: bool = False,
        mask_token_id: int = 151666,
        eos_token_id: int = 151645,
        pad_token_id: int = 151643,
        pad_target_penalty: float = 1.0,
        confidence_threshold: Optional[float] = None,
    ):

        '''
        block_decode_with_block_causal_attention_FreeDave (https://arxiv.org/abs/2510.00294) for block-causal models like SDAR and TraDo
        NOTE: now only consider 1 token per step; all the token transfer operations are performed in the block region. 
        '''

        batch_size = input_ids.shape[0]
        assert batch_size == 1, 'batch size must be 1 for FreeDave'

        prompt_length = input_ids.shape[1]
        num_total_blocks = (prompt_length + max_gen_length + block_length - 1) // block_length # pad the whole sequence to be divisible by block_length
        max_length = num_total_blocks * block_length
        x = F.pad(input_ids, (0, max_length - prompt_length), value=mask_token_id)

        trajectory_step_map = torch.full((batch_size, max_length), -torch.inf, device=x.device, dtype=torch.long)
        global_step = 0

        num_prefill_blocks = prompt_length // block_length
        prefill_length = num_prefill_blocks * block_length
        
        assert block_length % self.token_per_step == 0, 'block_length={} must be divisible by self.token_per_step={}'.format(block_length, self.token_per_step)
        steps_per_block = block_length // self.token_per_step

        attention_mask, position_ids = self.get_attention_mask_and_position_ids(
            attention_mask=attention_mask,
            max_length=max_length,
            block_length=block_length,
            num_blocks=num_total_blocks,
            attention_type='block_causal'
        )
        past_key_values = None

        # Prefilling stage
        if prefill_length > 0:
            cur_x = x[:, :prefill_length]
            cur_attn_mask = attention_mask[..., prefill_length, :prefill_length]
            if cur_attn_mask.dim() == 3:
                cur_attn_mask = cur_attn_mask[:, None, :, :]
            cur_position_ids = position_ids[:, :prefill_length]
            _, past_key_values, _ = self.get_processed_model_outputs(
                cur_x,
                attention_mask=cur_attn_mask,
                position_ids=cur_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                store_kv=True if use_cache else False
            )

        # Decoding stage
        global_step = 0
        for block_idx in range(num_prefill_blocks, num_total_blocks):

            # determine the number of future blocks
            if eager_acceptance_mode:
                num_future_blocks = min(math.ceil(draft_steps / steps_per_block), num_total_blocks - block_idx - 1)
                confidence_priority_shift = torch.arange(num_future_blocks, -1, -1, device=x.device).repeat_interleave(block_length)
            else:
                num_future_blocks = 0
                confidence_priority_shift = None

            current_block_start = block_idx * block_length
            current_block_end = current_block_start + (num_future_blocks + 1) * block_length
            current_window = slice(current_block_start, current_block_end)

            # NOTE: still use the block-causal attention mask, instead of just merge the current block and future blocks into a larger block with full attention mask
            current_block_attention_mask = attention_mask[..., current_block_start:current_block_end, :current_block_end] # (1, 1, current_block_end - current_block_start, current_block_end)
            if current_block_attention_mask.dim() == 3:
                current_block_attention_mask = current_block_attention_mask[:, None, :, :]
            current_block_position_ids = position_ids[:, current_block_start:current_block_end]

            current_block_x = x[:, current_block_start:current_block_end].clone() # (batch, current_block_end - current_block_start)
            current_block_mask_index = (current_block_x == mask_token_id)

            logits, _, _ = self.get_processed_model_outputs(
                current_block_x,
                attention_mask=current_block_attention_mask,
                position_ids=current_block_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=False
            )

            current_block_x0, current_block_confidence = self.get_mask_token_prediction_and_confidence(
                logits=logits,
                mask_index=current_block_mask_index,
                mask_token_id=mask_token_id,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
            
            # decoding loop for the current block
            while True:
            # for step in range(steps_per_block):

                current_block_mask_index = (current_block_x == mask_token_id)

                if num_total_blocks - block_idx > 1 and num_future_blocks > 0:
                    current_block_draft_steps = min(draft_steps, steps_per_block * num_future_blocks)
                else:
                    current_block_draft_steps = min(draft_steps, steps_per_block - step)

                current_block_x_draft = self.token_transfer(
                    x=current_block_x,
                    x0=current_block_x0,
                    draft_steps=current_block_draft_steps,
                    confidence=current_block_confidence,
                    confidence_priority_shift=confidence_priority_shift,
                    block_length=block_length,
                    mask_token_id=mask_token_id,
                    alg_temp=alg_temp,
                    token_per_step=self.token_per_step,
                ) # (batch_size * current_block_draft_steps, current_block_end - current_block_start)
                assert current_block_x_draft.shape == (batch_size * current_block_draft_steps, current_block_end - current_block_start), 'current_block_x_draft.shape must be (batch_size * current_block_draft_steps, current_block_end - current_block_start)'

                if num_total_blocks - block_idx == 1 and current_block_mask_index.sum() == self.token_per_step:
                    # if the last step in the last block, accept draft tokens and exit
                    # here draft tokens should be sampled by single step
                    # Take every interval-th (current_block_draft_steps-th) sample in batch as per interval batch index
                    current_block_x = current_block_x_draft[::current_block_draft_steps]
                    newly_unmasked = current_block_mask_index & (current_block_x != mask_token_id)
                    trajectory_step_map[:, current_block_start:current_block_end][newly_unmasked] = global_step
                    global_step += 1
                    step += 1
                    break
                else:
                    if not eager_acceptance_mode and current_block_mask_index.sum() == self.token_per_step:
                        # with eager mode disabled, accept draft tokens and exit at the last step in the current block
                        current_block_x = current_block_x_draft[::current_block_draft_steps]
                        newly_unmasked = current_block_mask_index & (current_block_x != mask_token_id)
                        trajectory_step_map[:, current_block_start:current_block_end][newly_unmasked] = global_step
                        global_step += 1
                        step += 1
                        break
                    else:
                        # batch forward the draft blocks
                        # v2: use tree attention (https://arxiv.org/abs/2401.10774) instead of expanding kv cache to further reduce memory overhead
                        current_draft_blocks_tree_attention_mask, current_draft_blocks_tree_position_ids = self.create_tree_attention_mask_and_position_ids(
                            attention_mask=current_block_attention_mask,
                            position_ids=current_block_position_ids,
                            shared_prefix_length=current_block_start,
                            branch_length=current_block_end - current_block_start,
                            shared_suffix_length=0,
                            num_branches=current_block_draft_steps
                        )

                        logits, _, _ = self.get_processed_model_outputs(
                            current_block_x_draft.view(batch_size, -1), # (batch_size, (current_block_end - current_block_start) * current_block_draft_steps)
                            attention_mask=current_draft_blocks_tree_attention_mask,
                            position_ids=current_draft_blocks_tree_position_ids,
                            past_key_values=past_key_values,
                            use_cache=True,
                            store_kv=False
                        ) # (batch_size, (current_block_end - current_block_start) * current_block_draft_steps, vocab_size)

                        current_block_x0_draft, current_block_confidence_draft = self.get_mask_token_prediction_and_confidence(
                            logits=logits.view(current_block_x_draft.shape[0], -1, logits.shape[-1]), # (batch_size * current_block_draft_steps, current_block_end - current_block_start, vocab_size)
                            mask_index=(current_block_x_draft == mask_token_id), # (batch_size * current_block_draft_steps, current_block_end - current_block_start)
                            mask_token_id=mask_token_id,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k
                        )

                        current_block_x_target = self.token_transfer(
                            x=current_block_x_draft,
                            x0=current_block_x0_draft,
                            confidence=current_block_confidence_draft,
                            confidence_priority_shift=confidence_priority_shift,
                            block_length=block_length,
                            mask_token_id=mask_token_id,
                            alg_temp=alg_temp,
                            token_per_step=self.token_per_step,
                        )
                        
                        # draft tokens verification
                        current_block_x_draft = current_block_x_draft.view(current_block_x.shape[0], current_block_draft_steps, *current_block_x.shape[1:]) # [batch_size, current_block_draft_steps, current_block_end - current_block_start]
                        current_block_x_target = current_block_x_target.view(current_block_x.shape[0], current_block_draft_steps, *current_block_x.shape[1:]) # [batch_size, current_block_draft_steps, current_block_end - current_block_start]
                        matched_draft_index = (current_block_x_target[:, :-1, :] == current_block_x_draft[:, 1:, :]).all(dim=-1)
                        matched_steps = torch.cumprod(matched_draft_index, dim=-1).sum(dim=-1).item() # assume batch size is 1

                        # Update current x, current kv cache, and reuse intermediate results
                        total_accepted_steps += matched_steps
                        total_draft_steps += current_block_draft_steps
                        if eager_acceptance_mode:
                            # TODO: implement eager acceptance mode
                            pass
                        else:
                            current_block_x = current_block_x_draft[:, matched_steps, :]
                            newly_unmasked = current_block_mask_index & (current_block_x != mask_token_id)
                            trajectory_step_map[:, current_block_start:current_block_end][newly_unmasked] = global_step
                            global_step += 1
                            current_block_x0 = current_block_x0_draft.view(batch_size, current_block_draft_steps, *current_block_x.shape[1:])[:, matched_steps, :]
                            current_block_confidence = current_block_confidence_draft.view(batch_size, current_block_draft_steps, *current_block_x.shape[1:])[:, matched_steps, :]

                        step += 1

                if (current_block_x[:, :block_length] == mask_token_id).all():
                    # if the current block is all unmasked, store the current block's kv and exit the decoding loop
                    _, past_key_values, _ = self.get_processed_model_outputs(
                        current_block_x[:, :block_length],
                        current_block_attention_mask[..., :block_length, :current_block_start + block_length],
                        current_block_position_ids[:, :block_length],
                        past_key_values=past_key_values,
                        use_cache=True,
                        store_kv=True
                    )
                    break

            # commit the current block to the sequence
            x[:, current_block_start:current_block_end] = current_block_x

        return x, trajectory_step_map[:, prompt_length:prompt_length + max_gen_length]