from transformers.generation import LogitsProcessor
from transformers import AutoTokenizer
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import warnings

from transformers.utils import add_start_docstrings

LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""

class ConstrainedLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        base_model: str = None,
        eos_token_id: int = None
    ):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams
        self.count=0
        self.base_model = base_model
        self.eos_token_id = eos_token_id
        if self.base_model.lower().find("gpt2") > -1:
            self.prefix_index = 4
        else:
            self.prefix_index = 3

    
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, float('-inf'))
            
        # Copy prefixes once per decoding step instead of synchronizing CUDA
        # separately for every beam. Apply all allowed coordinates in one write.
        width = self.prefix_index if self.count == 0 else self.count
        prefixes = input_ids[:, -width:].tolist()
        rows, tokens = [], []
        for row, hash_key in enumerate(prefixes):
            allowed = self._prefix_allowed_tokens_fn(row // self._num_beams, hash_key)
            if len(allowed) == 0:
                warnings.warn(
                    f"No valid tokens found for hash_key {hash_key} at step {self.count}. "
                    f"This indicates the model generated an unexpected token. "
                )
                if self.eos_token_id is not None:
                    allowed = [self.eos_token_id]
            rows.extend([row] * len(allowed))
            tokens.extend(allowed)
        if rows:
            row_ids = torch.tensor(rows, dtype=torch.long, device=scores.device)
            token_ids = torch.tensor(tokens, dtype=torch.long, device=scores.device)
            mask[row_ids, token_ids] = 0

        self.count += 1

        scores = scores + mask
        return scores
