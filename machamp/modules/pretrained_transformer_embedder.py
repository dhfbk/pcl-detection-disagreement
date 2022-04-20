import math
from typing import Optional, Tuple, List

from overrides import overrides

import torch
import torch.nn.functional as F
from transformers import XLNetConfig

from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data import Vocabulary
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.nn.util import batched_index_select
from allennlp.modules import Embedding
import logging
logger = logging.getLogger(__name__)

@TokenEmbedder.register("machamp_pretrained_transformer")
class MachampPretrainedTransformerEmbedder(TokenEmbedder):
    """
    Uses a pretrained model from `transformers` as a `TokenEmbedder`.

    Registered as a `TokenEmbedder` with name "pretrained_transformer".

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use. Should be the same as the corresponding
        `PretrainedTransformerIndexer`.
    max_length : `int`, optional (default = `None`)
        If positive, folds input token IDs into multiple segments of this length, pass them
        through the transformer model independently, and concatenate the final representations.
        Should be set to the same value as the `max_length` option on the
        `PretrainedTransformerIndexer`.
    sub_module: `str`, optional (default = `None`)
        The name of a submodule of the transformer to be used as the embedder. Some transformers naturally act
        as embedders such as BERT. However, other models consist of encoder and decoder, in which case we just
        want to use the encoder.
    train_parameters: `bool`, optional (default = `True`)
        If this is `True`, the transformer weights get updated during training.
    last_layer_only: `bool`, optional (default = `True`)
        When `True` (the default), only the final layer of the pretrained transformer is taken
        for the embeddings. But if set to `False`, a scalar mix of all of the layers
        is used.
    gradient_checkpointing: `bool`, optional (default = `None`)
        Enable or disable gradient checkpointing.
    """

    def __init__(
        self,
        model_name: str,
        max_length: int = None,
        sub_module: str = None,
        train_parameters: bool = True,
        layers_to_use: List[int] = [-1],
        override_weights_file: Optional[str] = None,
        override_weights_strip_prefix: Optional[str] = None,
        gradient_checkpointing: Optional[bool] = None,
    ) -> None:
        super().__init__()
        from allennlp.common import cached_transformers

        self.transformer_model = cached_transformers.get(
            model_name, True, override_weights_file, override_weights_strip_prefix
        )

        if gradient_checkpointing is not None:
            self.transformer_model.config.update({"gradient_checkpointing": gradient_checkpointing})

        self.config = self.transformer_model.config
        if sub_module:
            assert hasattr(self.transformer_model, sub_module)
            self.transformer_model = getattr(self.transformer_model, sub_module)
        self._max_length = max_length

        # I'm not sure if this works for all models; open an issue on github if you find a case
        # where it doesn't work.
        self.output_dim = self.config.hidden_size

        self.layers_to_use = layers_to_use
        self._scalar_mix: Optional[ScalarMix] = None
        if len(layers_to_use) > 1 or layers_to_use != [-1]:
            self.config.output_hidden_states = True
        if len(layers_to_use) > 1:
            self._scalar_mix = ScalarMix(len(layers_to_use))

        tokenizer = PretrainedTransformerTokenizer(model_name)
        self._num_added_start_tokens = len(tokenizer.single_sequence_start_tokens)
        self._num_added_end_tokens = len(tokenizer.single_sequence_end_tokens)
        self._num_added_tokens = self._num_added_start_tokens + self._num_added_end_tokens

        if not train_parameters:
            for param in self.transformer_model.parameters():
                param.requires_grad = False

    @overrides
    def get_output_dim(self):
        return self.output_dim

    def _number_of_token_type_embeddings(self):
        if isinstance(self.config, XLNetConfig):
            return 3  # XLNet has 3 type ids
        elif hasattr(self.config, "type_vocab_size"):
            return self.config.type_vocab_size
        else:
            return 0

    @overrides
    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor, 
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.BoolTensor] = None,
        dataset_ids: torch.LongTensor = None,
        wordpiece_sizes: List[List[int]] = None,
        dataset_embedder: Embedding= None
    ) -> torch.Tensor:  # type: ignore
        """
        # Parameters

        token_ids: `torch.LongTensor`
            Shape: `[batch_size, num_wordpieces if max_length is None else num_segment_concat_wordpieces]`.
            num_segment_concat_wordpieces is num_wordpieces plus special tokens inserted in the
            middle, e.g. the length of: "[CLS] A B C [SEP] [CLS] D E F [SEP]" (see indexer logic).
        mask: `torch.BoolTensor`
            Shape: [batch_size, num_wordpieces].
        type_ids: `Optional[torch.LongTensor]`
            Shape: `[batch_size, num_wordpieces if max_length is None else num_segment_concat_wordpieces]`.
        segment_concat_mask: `Optional[torch.BoolTensor]`
            Shape: `[batch_size, num_segment_concat_wordpieces]`.

        # Returns

        `torch.Tensor`
            Shape: `[batch_size, num_wordpieces, embedding_size]`.

        """
        # Some of the huggingface transformers don't support type ids at all and crash when you supply
        # them. For others, you can supply a tensor of zeros, and if you don't, they act as if you did.
        # There is no practical difference to the caller, so here we pretend that one case is the same
        # as another case.
        if type_ids is not None:
            max_type_id = type_ids.max()
            if max_type_id == 0:
                type_ids = None
            else:
                if max_type_id >= self._number_of_token_type_embeddings():
                    raise ValueError("Found type ids too large for the chosen transformer model.")
                assert token_ids.shape == type_ids.shape

        fold_long_sequences = self._max_length is not None and token_ids.size(1) > self._max_length
        if fold_long_sequences:
            batch_size, num_segment_concat_wordpieces = token_ids.size()
            token_ids, segment_concat_mask, type_ids = self._fold_long_sequences(
                token_ids, segment_concat_mask, type_ids
            )

        transformer_mask = segment_concat_mask if self._max_length is not None else mask
        # Shape: [batch_size, num_wordpieces, embedding_size],
        # or if self._max_length is not None:
        # [batch_size * num_segments, self._max_length, embedding_size]

        # ROB: here we search for rows to keep and to remove
        # rows with only 0's (<s>) and 2's (<pad>) should be removed.
        rm = []
        keep = []
        for sentIdx in range(len(token_ids)):
            allMask = set(transformer_mask[sentIdx].tolist())
            if len(allMask) == 1 and transformer_mask[sentIdx][0] == False:
                rm.append(sentIdx)
                rmMask = transformer_mask[sentIdx]
            else:
                keep.append(sentIdx)
        if len(keep) != len(token_ids):
            token_ids = token_ids[keep,:]
            transformer_mask = transformer_mask[keep, :]
        # We call this with kwargs because some of the huggingface models don't have the
        # token_type_ids parameter and fail even when it's given as None.
        # Also, as of transformers v2.5.1, they are taking FloatTensor masks.
        if dataset_ids == None:
            parameters = {"input_ids": token_ids, "attention_mask": transformer_mask.float()}
        else:# when using dataset_embeds, we override the word-embeddings
            parameters = {"attention_mask": transformer_mask.float()}

        if type_ids is not None:
            #parameters["token_type_ids"] = torch.ones_like(token_ids)
            parameters["token_type_ids"] = type_ids

        # As far as I can tell, the hidden states will always be the last element
        # in the output tuple as long as the model is not also configured to return
        # attention scores.
        # See, for example, the return value description for BERT:
        # https://huggingface.co/transformers/model_doc/bert.html#transformers.BertModel.forward
        # These hidden states will also include the embedding layer, which we don't
        # include in the scalar mix. Hence the `[1:]` slicing.
        
        # dataset embeds are being added here:
        if dataset_ids != None:

            word_embeds = self.transformer_model.embeddings.word_embeddings(token_ids)
            # Here we convert a list of tensors to something the same size as token_ids:
            # TODO, this can probably be done more efficient?

            # we have two things to use here:
            #wordpiece_sizes: list of lists of sizes of wordpieces
            #/dataset_ids: gold dataset_ids per word
            # we now have to expand the dataset ids, so that we have 1 per wordpiece 
            # instead of per word

            data_ids = torch.zeros_like(token_ids)
            for sent_idx in range(len(wordpiece_sizes)):
                piece_idx = 0
                for word_idx in range(min(len(wordpiece_sizes[sent_idx]), len(dataset_ids[sent_idx]))):
                    for _ in range(wordpiece_sizes[sent_idx][word_idx]):
                        if piece_idx >= len(dataset_ids[sent_idx]):
                            continue
                        data_ids[sent_idx][piece_idx] = dataset_ids[sent_idx][word_idx]
                        piece_idx += 1
            dataset_embeds = dataset_embedder(data_ids)
            # TODO could mask unknowns? (with token @@UNKNOWN@@)
            parameters['inputs_embeds'] = word_embeds+dataset_embeds

        transformer_output = self.transformer_model.forward(**parameters)
        if self.layers_to_use == [-1]:
            embeddings = transformer_output[0]
        elif len(self.layers_to_use) == 1:
            hidden_states = transformer_output[-1][1:]
            embeddings = hidden_states[self.layers_to_use[0]]
        else:
            hidden_states = torch.stack(transformer_output[-1][1:])
            selected_layers = hidden_states[self.layers_to_use]
            embeddings = self._scalar_mix(selected_layers)

        # ROB: Here we copy the embedding of row 0 to the index of the removed row, just 
        # to make it the same shape and to make sure everything > rmIdx still matches
        for idx in rm: 
            fullList = list(range(len(embeddings)))
            fullList.insert(idx, 0)
            embeddings = embeddings[fullList,:]

        if fold_long_sequences:
            embeddings = self._unfold_long_sequences(
                embeddings, segment_concat_mask, batch_size, num_segment_concat_wordpieces
            )

        return embeddings

    def _fold_long_sequences(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.LongTensor, torch.LongTensor, Optional[torch.LongTensor]]:
        """
        We fold 1D sequences (for each element in batch), returned by `PretrainedTransformerIndexer`
        that are in reality multiple segments concatenated together, to 2D tensors, e.g.

        [ [CLS] A B C [SEP] [CLS] D E [SEP] ]
        -> [ [ [CLS] A B C [SEP] ], [ [CLS] D E [SEP] [PAD] ] ]
        The [PAD] positions can be found in the returned `mask`.

        # Parameters

        token_ids: `torch.LongTensor`
            Shape: `[batch_size, num_segment_concat_wordpieces]`.
            num_segment_concat_wordpieces is num_wordpieces plus special tokens inserted in the
            middle, i.e. the length of: "[CLS] A B C [SEP] [CLS] D E F [SEP]" (see indexer logic).
        mask: `torch.BoolTensor`
            Shape: `[batch_size, num_segment_concat_wordpieces]`.
            The mask for the concatenated segments of wordpieces. The same as `segment_concat_mask`
            in `forward()`.
        type_ids: `Optional[torch.LongTensor]`
            Shape: [batch_size, num_segment_concat_wordpieces].

        # Returns:

        token_ids: `torch.LongTensor`
            Shape: [batch_size * num_segments, self._max_length].
        mask: `torch.BoolTensor`
            Shape: [batch_size * num_segments, self._max_length].
        """
        num_segment_concat_wordpieces = token_ids.size(1)
        num_segments = math.ceil(num_segment_concat_wordpieces / self._max_length)
        padded_length = num_segments * self._max_length
        length_to_pad = padded_length - num_segment_concat_wordpieces

        def fold(tensor):  # Shape: [batch_size, num_segment_concat_wordpieces]
            # Shape: [batch_size, num_segments * self._max_length]
            tensor = F.pad(tensor, [0, length_to_pad], value=0)
            # Shape: [batch_size * num_segments, self._max_length]
            return tensor.reshape(-1, self._max_length)

        return fold(token_ids), fold(mask), fold(type_ids) if type_ids is not None else None

    def _unfold_long_sequences(
        self,
        embeddings: torch.FloatTensor,
        mask: torch.BoolTensor,
        batch_size: int,
        num_segment_concat_wordpieces: int,
    ) -> torch.FloatTensor:
        """
        We take 2D segments of a long sequence and flatten them out to get the whole sequence
        representation while remove unnecessary special tokens.

        [ [ [CLS]_emb A_emb B_emb C_emb [SEP]_emb ], [ [CLS]_emb D_emb E_emb [SEP]_emb [PAD]_emb ] ]
        -> [ [CLS]_emb A_emb B_emb C_emb D_emb E_emb [SEP]_emb ]

        We truncate the start and end tokens for all segments, recombine the segments,
        and manually add back the start and end tokens.

        # Parameters

        embeddings: `torch.FloatTensor`
            Shape: [batch_size * num_segments, self._max_length, embedding_size].
        mask: `torch.BoolTensor`
            Shape: [batch_size * num_segments, self._max_length].
            The mask for the concatenated segments of wordpieces. The same as `segment_concat_mask`
            in `forward()`.
        batch_size: `int`
        num_segment_concat_wordpieces: `int`
            The length of the original "[ [CLS] A B C [SEP] [CLS] D E F [SEP] ]", i.e.
            the original `token_ids.size(1)`.

        # Returns:

        embeddings: `torch.FloatTensor`
            Shape: [batch_size, self._num_wordpieces, embedding_size].
        """

        def lengths_to_mask(lengths, max_len, device):
            return torch.arange(max_len, device=device).expand(
                lengths.size(0), max_len
            ) < lengths.unsqueeze(1)

        device = embeddings.device
        num_segments = int(embeddings.size(0) / batch_size)
        embedding_size = embeddings.size(2)

        # We want to remove all segment-level special tokens but maintain sequence-level ones
        num_wordpieces = num_segment_concat_wordpieces - (num_segments - 1) * self._num_added_tokens

        embeddings = embeddings.reshape(batch_size, num_segments * self._max_length, embedding_size)
        mask = mask.reshape(batch_size, num_segments * self._max_length)
        # We assume that all 1s in the mask precede all 0s, and add an assert for that.
        # Open an issue on GitHub if this breaks for you.
        # Shape: (batch_size,)
        seq_lengths = mask.sum(-1)
        if not (lengths_to_mask(seq_lengths, mask.size(1), device) == mask).all():
            raise ValueError(
                "Long sequence splitting only supports masks with all 1s preceding all 0s."
            )
        # Shape: (batch_size, self._num_added_end_tokens); this is a broadcast op
        end_token_indices = (
            seq_lengths.unsqueeze(-1) - torch.arange(self._num_added_end_tokens, device=device) - 1
        )

        # Shape: (batch_size, self._num_added_start_tokens, embedding_size)
        start_token_embeddings = embeddings[:, : self._num_added_start_tokens, :]
        # Shape: (batch_size, self._num_added_end_tokens, embedding_size)
        end_token_embeddings = batched_index_select(embeddings, end_token_indices)

        embeddings = embeddings.reshape(batch_size, num_segments, self._max_length, embedding_size)
        embeddings = embeddings[
            :, :, self._num_added_start_tokens : -self._num_added_end_tokens, :
        ]  # truncate segment-level start/end tokens
        embeddings = embeddings.reshape(batch_size, -1, embedding_size)  # flatten

        # Now try to put end token embeddings back which is a little tricky.

        # The number of segment each sequence spans, excluding padding. Mimicking ceiling operation.
        # Shape: (batch_size,)
        num_effective_segments = (seq_lengths + self._max_length - 1) // self._max_length
        # The number of indices that end tokens should shift back.
        num_removed_non_end_tokens = (
            num_effective_segments * self._num_added_tokens - self._num_added_end_tokens
        )
        # Shape: (batch_size, self._num_added_end_tokens)
        end_token_indices -= num_removed_non_end_tokens.unsqueeze(-1)
        assert (end_token_indices >= self._num_added_start_tokens).all()
        # Add space for end embeddings
        embeddings = torch.cat([embeddings, torch.zeros_like(end_token_embeddings)], 1)
        # Add end token embeddings back
        embeddings.scatter_(
            1, end_token_indices.unsqueeze(-1).expand_as(end_token_embeddings), end_token_embeddings
        )

        # Now put back start tokens. We can do this before putting back end tokens, but then
        # we need to change `num_removed_non_end_tokens` a little.
        embeddings = torch.cat([start_token_embeddings, embeddings], 1)

        # Truncate to original length
        embeddings = embeddings[:, :num_wordpieces, :]
        return embeddings
