import torch

from torch.nn import functional as F
from torch import nn

from stable_audio_tools.training.losses import LossModule

# https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
class CTCLossModule(LossModule):
    def __init__(
        self,
        name: str,
        input_key: str,
        target_key: str,
        weight: float = 1.0,
        decay: float = 1.0,
        blank_idx: int = 0,
        padding_idx: int = None,
        input_lengths_key: str = None,
    ):
        super().__init__(name=name, weight=weight, decay=decay)
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)
        self.input_key = input_key
        self.target_key = target_key
        self.input_lengths_key = input_lengths_key
        self.blank_idx = blank_idx
        self.padding_idx = padding_idx if padding_idx is not None else blank_idx + 1

    def forward(self, info):
        """
        Computes the CTC loss.

        Args:
            info (dict): Dictionary containing model outputs and other relevant data.
                - info[self.input_key]: Model logits of shape (batch_size, sequence_length, num_classes).
                - info[self.target_key]: Target data (list of dicts with 'phone' key).
                - info[self.input_lengths_key]: (Optional) Actual lengths of the input sequences.

        Returns:
            loss (Tensor): The computed CTC loss, scaled by the weight.
        """
        # Build targets and target lengths
        padded_targets, target_lengths = build_target(info[self.target_key], self.padding_idx)

        # Get logits from the model output
        logits = info[self.input_key]  # Expected shape: (batch_size, sequence_length, num_classes)

        # Move logits to the device of phonemes
        device = padded_targets.device
        logits = logits.to(device)

        # Apply log_softmax to obtain log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # Shape: (batch_size, seq_length, num_classes)

        # Transpose log_probs to match (seq_length, batch_size, num_classes)
        log_probs = log_probs.permute(1, 0, 2)  # Now shape is (seq_length, batch_size, num_classes)

        # Determine input lengths
        if self.input_lengths_key and self.input_lengths_key in info:
            input_lengths = info[self.input_lengths_key].to(device)
        else:
            # Assume all input sequences have the same length
            input_lengths = torch.full(
                (log_probs.size(1),),  # batch_size
                log_probs.size(0),     # seq_length
                dtype=torch.long,
                device=device
            )

        # Compute the CTC loss
        loss = self.ctc_loss(log_probs, padded_targets, input_lengths, target_lengths)

        loss = self.weight * loss

        return loss

class PERModule(nn.Module):
    def __init__(
        self,
        input_key: str,
        target_key: str,
        blank_idx: int = 0,
        padding_idx: int = None,
    ):
        super().__init__()
        self.input_key = input_key
        self.target_key = target_key
        self.blank_idx = blank_idx
        self.padding_idx = padding_idx if padding_idx is not None else blank_idx + 1

    def decode_predictions(self, predicted_ids):
        """
        Decodes the model predictions by collapsing repeats and removing blanks.

        Args:
            predicted_ids (Tensor): Tensor of shape (seq_length,) containing predicted token IDs.

        Returns:
            List[int]: Decoded sequence of token IDs.
        """
        predicted_sequence = []
        previous_id = None
        for id in predicted_ids:
            id = id.item()
            if id != self.blank_idx and id != previous_id:
                predicted_sequence.append(id)
            previous_id = id
        return predicted_sequence

    def forward(self, info):
        """
        Computes the CTC loss.

        Args:
            info (dict): Dictionary containing model outputs and other relevant data.
                - info[self.input_key]: Model logits of shape (batch_size, sequence_length, num_classes).
                - info[self.target_key]: Target data (list of dicts with 'phone' key).
                - info[self.input_lengths_key]: (Optional) Actual lengths of the input sequences.

        Returns:
            loss (Tensor): The computed CTC loss, scaled by the weight.
        """
        with torch.no_grad():
            # Build targets and target lengths
            padded_targets, target_lengths = build_target(info[self.target_key], self.padding_idx)

            # Get logits from the model output
            logits = info[self.input_key]  # Expected shape: (batch_size, sequence_length, num_classes)

            # Move logits to the device of phonemes
            device = padded_targets.device
            logits = logits.to(device)

            # Apply log_softmax to obtain log probabilities
            log_probs = F.log_softmax(logits, dim=-1)  # Shape: (batch_size, seq_length, num_classes)

            # Transpose log_probs to match (seq_length, batch_size, num_classes)
            log_probs = log_probs.permute(1, 0, 2)  # Now shape is (seq_length, batch_size, num_classes)

            # Get predictions via greedy decoding
            predicted_ids = torch.argmax(logits, dim=-1)  # Shape: (batch_size, seq_length)

            batch_size = predicted_ids.size(0)
            pers = []

            for i in range(batch_size):
                # Decode predictions
                pred_ids = predicted_ids[i]  # Tensor of shape (seq_length,)
                pred_sequence = self.decode_predictions(pred_ids)

                # Get target sequence
                target_ids = padded_targets[i]  # Tensor of shape (max_target_length,)
                target_length = target_lengths[i]
                target_sequence = target_ids[:target_length].tolist()

                # Remove padding tokens from target sequence
                target_sequence = [id for id in target_sequence if id != self.padding_idx]

                # Compute edit distance using the editdistance package
                # distance = editdistance.eval(pred_sequence, target_sequence)
                distance = edit_distance(pred_sequence, target_sequence)

                # Compute PER
                per = distance / max(len(target_sequence), 1)
                pers.append(per)

            # Compute average PER over the batch
            average_per = sum(pers) / len(pers)

            return average_per

def edit_distance(seq1, seq2):
        """
        Computes the edit distance between two sequences.

        Args:
            seq1 (List[int]): First sequence.
            seq2 (List[int]): Second sequence.

        Returns:
            int: The edit distance between seq1 and seq2.
        """
        m = len(seq1)
        n = len(seq2)
        # Create a DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        # Initialize
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        # Compute dp table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    cost = 0
                else:
                    cost = 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # deletion
                    dp[i][j - 1] + 1,      # insertion
                    dp[i - 1][j - 1] + cost  # substitution
                )
        return dp[m][n]

def build_target(batch, padding_idx):
    """
    Builds padded targets and computes target lengths.

    Args:
        batch (list): A list of dictionaries, each containing a 'phone' key with tensor values.

    Returns:
        padded_targets (Tensor): Padded target sequences of shape (batch_size, max_target_length).
        target_lengths (Tensor): Lengths of each target sequence in the batch.
    """
    # Extract phoneme sequences
    phoneme_sequences = [item['phone'] for item in batch]

    # Determine device from the phoneme sequences
    device = phoneme_sequences[0].device

    # Ensure phoneme sequences are 1D tensors
    phoneme_sequences = [seq.view(-1) if seq.ndim > 1 else seq for seq in phoneme_sequences]

    # Compute target lengths
    target_lengths = torch.tensor([seq.size(0) for seq in phoneme_sequences], dtype=torch.long, device=device)

    # Pad sequences
    padded_targets = nn.utils.rnn.pad_sequence(
        phoneme_sequences,
        batch_first=True,
        padding_value=padding_idx
    ).to(device)

    return padded_targets, target_lengths
