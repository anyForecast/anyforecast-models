import collections

import torch

from deepts.utils import rnn


class InputDict(dict):
    pass


class Seq2SeqCollateFn:
    """Customized Seq2Seq collate function."""

    def pad_sequences(
        self, x_dict: dict[str, list], keys: tuple
    ) -> dict[str, list | torch.Tensor]:
        """Pads sequences.

        Parameters
        ----------
        x_dict : dict str -> list
            dict containing the lists to be padded.

        keys : tuple of str
            Keys referencing the lists to be padded.

        Warnings
        --------
        In-place operation.
        """
        padded_sequences = {
            k: rnn.pad_sequence(x_dict[k], batch_first=True) for k in keys
        }

        x_dict.update(padded_sequences)
        return x_dict

    def update_dict(
        self, x_dict: dict[str, list], sample: tuple
    ) -> dict[str, list]:
        """Updates dict with sample data.

        Warnings
        --------
        In-place operation.
        """
        X, y = sample
        length = X["encoder_length"]

        x_dict["encoder_lengths"].append(length)
        x_dict["decoder_lengths"].append(X["decoder_length"])
        x_dict["categoricals"].append(X["x_cat"][0])
        x_dict["encoder_cont"].append(X["x_cont"][:length])
        x_dict["decoder_cont"].append(X["x_cont"][length:])

        # TODO: Handle multi-target case.
        x_dict["decoder_target"].append(y[0])

        return x_dict

    def __call__(
        self, batch: list[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> tuple[dict[str, InputDict], torch.Tensor]:
        """Collate function to combine items into mini-batch for dataloader.

        Parameters
        ----------
        batch : list[tuple[dict[str, torch.Tensor], torch.Tensor]]:
            list of samples.

        Returns
        -------
        minibatch : tuple[dict[str, InputDict], torch.Tensor]
        """
        sequences_to_pad = ("encoder_cont", "decoder_cont", "decoder_target")

        x_dict: InputDict = collections.defaultdict(list)
        for sample in batch:
            x_dict = self.update_dict(x_dict, sample)

        # Pad sequences to handle variable length Tensors.
        x_dict = self.pad_sequences(x_dict, sequences_to_pad)

        # Convert non-padded lists to Tensors.
        x_dict["categoricals"] = torch.stack(x_dict["categoricals"])
        x_dict["encoder_lengths"] = torch.LongTensor(x_dict["encoder_lengths"])
        x_dict["decoder_lengths"] = torch.LongTensor(x_dict["decoder_lengths"])

        X = {"x": x_dict}
        y = x_dict["decoder_target"]

        return X, y
