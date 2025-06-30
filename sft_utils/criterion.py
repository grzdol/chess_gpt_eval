from nanogpt.criterion import Criterion
import torch
import torch.nn.functional as F

class SFTCriterion(Criterion):
  def __init__(self, trained_model, ref_model, alpha=10):
    self.alpha = alpha
    self.trained_model = trained_model
    self.ref_model = ref_model
    self.device = next(trained_model.parameters()).device
    
  @staticmethod
  def calc_loss(y_pred, y_seqs, mask_rows):
    """
    y_pred:  Tensor of shape (B, L, V) — logits over V classes
    y_seqs:  LongTensor of shape (B, L) — true class indices
    mask_rows: ByteTensor (or BoolTensor) of shape (B, L) — 1 where we want to include in the loss

    Returns:
      scalar loss = average cross‐entropy over all masked positions
    """
    B, L, V = y_pred.shape

    # flatten everything to shape (B*L, ...)
    logits_flat = y_pred.view(-1, V)            # → (B*L, V)
    target_flat = y_seqs.view(-1)               # → (B*L,)
    mask_flat   = mask_rows.view(-1).bool()     # → (B*L,)

    # if nothing to compute (all truncated), return zero
    if mask_flat.sum() == 0:
        return torch.tensor(0., device=y_pred.device, requires_grad=True)

    # select only the masked entries
    logits_masked = logits_flat[mask_flat]       # → (M, V)
    target_masked = target_flat[mask_flat]       # → (M,)

    # compute CE; reduction='mean' → average over those M positions
    loss = F.cross_entropy(logits_masked, target_masked, reduction='mean')
    return loss
  
  @staticmethod
  def calc_kl_loss(log_p_posterior, log_p_prior, ctx_mask):
    """
    log_p_posterior: FloatTensor (B, L, V) — log-probs from your posterior
    log_p_prior:     FloatTensor (B, L, V) — log-probs from your prior
    ctx_mask:        ByteTensor  (B, L)   — 1 for context tokens

    Returns:
      scalar: average KL over context positions
    """
    # KL per token = sum_q p_post(q) * [log p_post(q) - log p_prior(q)]
    # F.kl_div takes (input, target) where input is log-probs w.r.t. target
    # so swap args and use reduction='none'
    kl_tensor = F.kl_div(
        log_p_prior,          # input: log p_prior
        log_p_posterior,      # target:   p_posterior (expects log-target)
        reduction='none',
        log_target=True
    )  # → (B, L, V)

    # sum over vocab to get per‐token KL
    kl_per_token = kl_tensor.sum(-1)  # → (B, L)

    mask = ctx_mask.bool().view(-1)   # flatten
    kl_flat = kl_per_token.view(-1)

    # if no context, avoid zero‐division
    if mask.sum() == 0:
        return torch.tensor(0., device=kl_per_token.device, requires_grad=True)

    return kl_flat[mask].mean()
  
  def get_loss(self, batch):
    X, Y, mask, ctx_mask = batch
    X = torch.tensor(X, device=self.device)
    Y = torch.tensor(Y, device=self.device)
    mask = mask.to(self.device)
    ctx_mask = ctx_mask.to(self.device)
    logits, _ = self.trained_model(X, Y)
    with torch.no_grad():
        prior_logits, _ = self.ref_model(X, Y)
    loss = self.calc_loss(logits, Y, mask)
    loss += self.alpha * self.calc_kl_loss(F.log_softmax(logits, dim=-1), F.log_softmax(prior_logits, dim=-1), ctx_mask)
    return loss