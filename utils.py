import re
import torch
import torch.nn.functional as F

def reformat_pgn_line(line: str) -> str:
    line = re.sub(r"\d+\.{3}\s*", "", line)     # drop black-move numbers
    line = re.sub(r"(\d+)\.\s+", r"\1.", line)  # remove space after white numbers
    return ';' + line   # collapse multiple spaces
  
  
def make_collate(tokenizer):
  return lambda batch: collate_fn(batch, tokenizer)
  
def collate_fn(batch, tokenizer, global_max_chars=1024):
    """
    batch: List[dict] where each dict has keys
       'id', 'ctx' (string), 'target' (string)
    tokenizer: callable mapping a string -> List[int]
    Returns:
       X_seqs:   List[List[int]]       # each inner list len = batch_max-1
       Y_seqs:   List[List[int]]       # each inner list len = batch_max-1
       loss_mask: torch.ByteTensor     # shape (B, batch_max-1)
    """
    pad_char = tokenizer(';')[0]

    # 1) tokenize inputs
    ctxs = [tokenizer(reformat_pgn_line(item['ctx'])) for item in batch]
    tgts = [tokenizer(item['target'].strip())      for item in batch]

    seqs      = []
    ctx_lens  = []
    orig_lens = []
    valid_for_loss = []

    # 2) build raw sequences, truncate to global_max_chars if needed
    for ctx_ids, tgt_ids in zip(ctxs, tgts):
        ctx_len = len(ctx_ids)
        seq = list(ctx_ids) + list(tgt_ids)
        orig_len = len(seq)

        if orig_len > global_max_chars:
            seq = seq[:global_max_chars]
            valid_for_loss.append(False)
        else:
            valid_for_loss.append(True)

        ctx_lens.append(ctx_len)
        orig_lens.append(orig_len)
        seqs.append(seq)

    # 3) find batch‐max (longest seq in this batch)
    batch_max = max(len(s) for s in seqs)

    X_seqs = []
    Y_seqs = []
    mask_rows = []

    # 4) pad all seqs to batch_max, build X/Y and loss‐mask
    for seq, ctx_len, orig_len, usable in zip(seqs, ctx_lens, orig_lens, valid_for_loss):
        # pad up to batch_max
        pad_len = batch_max - len(seq)
        if pad_len > 0:
            seq = seq + [pad_char] * pad_len

        # shift for model input/target
        X = seq[:-1]   # length = batch_max-1
        Y = seq[1:]    # length = batch_max-1

        # build loss mask
        if not usable:
            mask = [0] * (batch_max - 1)
        else:
            mask = []
            for i in range(batch_max - 1):
                idx = i + 1
                # only mask true-target positions
                mask.append(1 if (idx >= ctx_len and idx < orig_len) else 0)

        X_seqs.append(X)
        Y_seqs.append(Y)
        mask_rows.append(mask)

    # convert mask to ByteTensor
    loss_mask = torch.ByteTensor(mask_rows)
    return X_seqs, Y_seqs, loss_mask
  
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
  