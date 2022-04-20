from typing import Dict, Optional

import sys
import numpy
import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TimeDistributed
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure
from overrides import overrides
from torch.nn.modules.linear import Linear


@Model.register("machamp_tag_decoder")
class MachampTagger(Model):
    """
    This `SimpleTagger` simply encodes a sequence of text with a stacked `Seq2SeqEncoder`, then
    predicts a tag for each token in the sequence.

    Registered as a `Model` with name "simple_tagger".

    # Parameters

    vocab : `Vocabulary`, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : `TextFieldEmbedder`, required
        Used to embed the `tokens` `TextField` we get as input to the model.
    calculate_span_f1 : `bool`, optional (default=`None`)
        Calculate span-level F1 metrics during training. If this is `True`, then
        `label_encoding` is required. If `None` and
        label_encoding is specified, this is set to `True`.
        If `None` and label_encoding is not specified, it defaults
        to `False`.
    label_encoding : `str`, optional (default=`None`)
        Label encoding to use when calculating span f1.
        Valid options are "BIO", "BIOUL", "IOB1", "BMES".
        Required if `calculate_span_f1` is true.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.
    """

    def __init__(
        self,
        task: str,
        vocab: Vocabulary,
        input_dim: int,
        loss_weight: float= 1.0,
        training_dynamics: bool = False,
        metric: str = 'acc',
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self.task = task
        self.vocab = vocab
        self.loss_weight = loss_weight
        self.training_dynamics = training_dynamics
        self.input_dim = input_dim
        self.num_classes = self.vocab.get_vocab_size(task)
        self.tag_projection_layer = TimeDistributed(
            Linear(self.input_dim, self.num_classes)
        )
        if metric == "acc":
            self.metrics = {"acc": CategoricalAccuracy()}
        elif metric == "span_f1":
            print(f"To use \"{metric}\", please use the \"seq_bio\" decoder instead.")
            sys.exit()
        elif metric == "multi_span_f1":
            print(f"To use \"{metric}\", please use the \"multiseq\" decoder instead.")
            sys.exit()
        elif metric == "micro-f1":
            self.metrics = {"micro-f1": FBetaMeasure(average='micro')}
        elif metric == "macro-f1":
            self.metrics = {"macro-f1": FBetaMeasure(average='macro')}
        else:
            print(f"ERROR. Metric \"{metric}\" unrecognized. Using accuracy \"acc\" instead.")
            self.metrics = {"acc": CategoricalAccuracy()}


    @overrides
    def forward(
        self,  # type: ignore
        embedded_text: torch.LongTensor,
        gold_labels: torch.LongTensor = None,
        mask: torch.LongTensor = None,
        ignore_loss_on_o_tags: bool = False,
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters

        tokens : `TextFieldTensors`, required
            The output of `TextField.as_array()`, which should typically be passed directly to a
            `TextFieldEmbedder`. This output is a dictionary mapping keys to `TokenIndexer`
            tensors.  At its most basic, using a `SingleIdTokenIndexer` this is : `{"tokens":
            Tensor(batch_size, num_tokens)}`. This dictionary will have the same keys as were used
            for the `TokenIndexers` when you created the `TextField` representing your
            sequence.  The dictionary is designed to be passed directly to a `TextFieldEmbedder`,
            which knows how to combine different word representations into a single vector per
            token in your input.
        tags : `torch.LongTensor`, optional (default = `None`)
            A torch tensor representing the sequence of integer gold class labels of shape
            `(batch_size, num_tokens)`.
        ignore_loss_on_o_tags : `bool`, optional (default = `False`)
            If True, we compute the loss only for actual spans in `tags`, and not on `O` tokens.
            This is useful for computing gradients of the loss on a _single span_, for
            interpretation / attacking.

        # Returns

        An output dictionary consisting of:
            - `logits` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_tokens, tag_vocab_size)` representing
                unnormalised log probabilities of the tag classes.
            - `class_probabilities` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_tokens, tag_vocab_size)` representing
                a distribution of the tag classes per word.
            - `loss` (`torch.FloatTensor`, optional) :
                A scalar loss to be optimised.

        """
        batch_size, sequence_length, _ = embedded_text.shape

        logits = self.tag_projection_layer(embedded_text)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view(
            [batch_size, sequence_length, self.num_classes]
        )

        output_dict = {"logits": logits, "class_probabilities": class_probabilities, "log_training_dynamics": self.training_dynamics}

        if gold_labels is not None:
            if ignore_loss_on_o_tags:
                o_tag_index = self.vocab.get_token_index("O", namespace=self.task)
                tag_mask = mask & (gold_labels != o_tag_index)
            else:
                tag_mask = mask
            output_dict["loss"] = sequence_cross_entropy_with_logits(logits, gold_labels, tag_mask) * self.loss_weight
            for metric in self.metrics.values():
                metric(class_probabilities, gold_labels, mask)
        return output_dict

    @overrides
    def make_output_human_readable(
        self, predictions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a `"tags"` key to the dictionary with the result.
        """
        all_predictions = predictions.cpu().data.numpy()
        if all_predictions.ndim == 3:
            predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
        else:
            predictions_list = [all_predictions]

        all_tags = []
        for predictions in predictions_list:
            argmax_indices = numpy.argmax(predictions, axis=-1)
            tags = [
                self.vocab.get_token_from_index(x, namespace=self.task)
                for x in argmax_indices
            ]
            all_tags.append(tags)
        return all_tags

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        main_metrics = {}
        for metric_name, metric in self.metrics.items():
            if metric_name.endswith('f1'):
                if metric._true_positive_sum == None:
                    main_metrics[f".run/{self.task}/{metric_name}"] = {'precision':0.0, 'recall': 0.0, 'fscore': 0.0}
                else:
                    main_metrics[f".run/{self.task}/{metric_name}"] = metric.get_metric(reset)
            else:
                main_metrics[f".run/{self.task}/{metric_name}"] = metric.get_metric(reset)
        return {**main_metrics}

