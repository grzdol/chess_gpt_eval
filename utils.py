import re
import torch
import torch.nn.functional as F

def reformat_pgn_line(line: str) -> str:
    line = re.sub(r"\d+\.{3}\s*", "", line)     # drop black-move numbers
    line = re.sub(r"(\d+)\.\s+", r"\1.", line)  # remove space after white numbers
    return ';' + line   # collapse multiple spaces
  
  
def make_collate(tokenizer):
  return lambda batch: collate_fn(batch, tokenizer)
  
def collate_fn(batch, tokenizer):
    """
    batch: List[dict] where each dict has keys
       'id', 'ctx' (string), 'target' (string)
    Returns:
       X_seqs:   List[List[str]]       # each inner list len = 767
       Y_seqs:   List[List[str]]       # each inner list len = 767
       loss_mask: torch.ByteTensor     # shape (B, 767), 1 over true-target chars
    """
    max_chars = 768
    pad_char  = tokenizer(';')[0]
    X_seqs = []
    Y_seqs = []
    mask_rows = []

    # first, extract and clean all ctx / target strings
    ctxs = [tokenizer(reformat_pgn_line(item['ctx'])) for item in batch]
    tgts = [tokenizer(item['target'].strip())  for item in batch]

    for ctx_str, tgt_str in zip(ctxs, tgts):
        ctx_chars = list(ctx_str)
        tgt_chars = list(tgt_str)

        orig_len = len(ctx_chars) + len(tgt_chars)
        seq = ctx_chars + tgt_chars

        # truncate or pad to exactly max_chars
        if len(seq) > max_chars:
            seq = seq[:max_chars]
            use_for_loss = False
        else:
            seq += [pad_char] * (max_chars - len(seq))
            use_for_loss = True

        # build X and Y (shifted by one)
        X = seq[:-1]  # length = max_chars-1
        Y = seq[1:]   # length = max_chars-1

        # build loss mask: 1 only for true-target chars
        if not use_for_loss:
            # if truncated, we won’t compute loss
            mask = [0] * (max_chars - 1)
        else:
            mask = []
            for i in range(len(Y)):
                idx = i + 1
                # mark positions that fall in the original target span
                in_target = (idx >= len(ctx_chars)
                             and idx < orig_len)
                mask.append(1 if in_target else 0)

        X_seqs.append(X)
        Y_seqs.append(Y)
        mask_rows.append(mask)

    
    return X_seqs, Y_seqs, mask_rows
  
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
  