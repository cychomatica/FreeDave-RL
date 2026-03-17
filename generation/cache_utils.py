from transformers.cache_utils import DynamicCache
import torch
from typing import Optional, Dict, Any, Tuple

class DynamicDualCache(DynamicCache):
    def __init__(self):
        super().__init__()
        self.replace_position = None

    def set_replace_position(self, replace_position: torch.Tensor):
        self.replace_position = replace_position

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        if layer_idx == 0 and self.replace_position is None:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([]))
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            elif (
                not self.key_cache[layer_idx].numel()  # prefers not t.numel() to len(t) == 0 to export the model
            ):  # fills previously skipped layers; checking for tensor causes errors
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                if self.replace_position is not None:
                    replace_indices = self.replace_position.nonzero(as_tuple=True)[1]
                    self.key_cache[layer_idx][:, :, replace_indices, :] = key_states
                    self.value_cache[layer_idx][:, :, replace_indices, :] = value_states

        return self.key_cache[layer_idx], self.value_cache[layer_idx]