import torch
import torch.nn.functional as F
from torchtune.rlhf.loss import DPOLoss

class DPOCriterion:
    def __init__(self, trained_model, ref_model, beta=0.5, label_smoothing=0.0):
        """
        Args:
            trained_model: your fine-tunable policy model (nn.Module).
            ref_model:      a frozen reference copy of the model.
            beta:           temperature for DPO loss (0.1–0.5 typical).
            label_smoothing: uncertainty parameter (usually 0).
        """
        self.trained_model = trained_model
        self.ref_model     = ref_model
        self.loss_fn       = DPOLoss(beta=beta, label_smoothing=label_smoothing)

    def _get_log_probs(self, model, input_ids, attention_mask):
        """
        Compute sequence log-probabilities by summing per-token log-probs.
        """
        device = next(model.parameters()).device
        logits, _ = model(input_ids.to(device), attn_mask=attention_mask.to(device)) # [B, L, V]

        shift_logits = logits[:, :-1, :].contiguous().to(device)
        shift_labels = input_ids[:, 1:].contiguous().to(device)
        shift_mask   = attention_mask[:, 1:].contiguous().to(device)

        log_probs = F.log_softmax(shift_logits, dim=-1)                        # [B, L-1, V]
        token_lp  = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
        seq_logp  = (token_lp * shift_mask).sum(dim=-1)                       # [B]
        return seq_logp

    def get_loss(self, batch):
        """
        batch must contain:
          - chosen_input_ids, chosen_attention_mask
          - rejected_input_ids, rejected_attention_mask
        All are LongTensors of shape [B, L].
        """
        # policy log-probs
        policy_chosen_lp  = self._get_log_probs(self.trained_model,
                                                batch["chosen_input_ids"],
                                                batch["chosen_attention_mask"])
        policy_rejected_lp = self._get_log_probs(self.trained_model,
                                                 batch["rejected_input_ids"],
                                                 batch["rejected_attention_mask"])

        with torch.no_grad():
          ref_chosen_lp  = self._get_log_probs(self.ref_model,
                                               batch["chosen_input_ids"],
                                               batch["chosen_attention_mask"])
          ref_rejected_lp = self._get_log_probs(self.ref_model,
                                                batch["rejected_input_ids"],
                                                batch["rejected_attention_mask"])

        losses, chosen_rewards, rejected_rewards = self.loss_fn(
            policy_chosen_lp,
            policy_rejected_lp,
            ref_chosen_lp,
            ref_rejected_lp,
        )  # returns (losses[B], chosen_rewards[B], rejected_rewards[B]) :contentReference[oaicite:0]{index=0}

        # return mean training loss (you can optionally also return rewards or individual losses)
        return losses.mean()
