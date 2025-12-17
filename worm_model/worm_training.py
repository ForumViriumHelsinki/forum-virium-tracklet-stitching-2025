import torch
import torch.nn as nn


def train_one_epoch(
    model_enc, model_cls, loader, optimizer, device, lam=0.2, margin=0.2
):
    bce = nn.BCEWithLogitsLoss()
    trip = nn.TripletMarginLoss(margin=margin)

    model_enc.train()
    model_cls.train()

    total = 0.0
    for batch in loader:
        loc_a = batch["loc_a"].to(device)  # (B,5,2)
        meta_a = batch["meta_a"].to(device)  # (B,3)
        loc_b = batch["loc_b"].to(device)
        meta_b = batch["meta_b"].to(device)
        loc_n = batch["loc_n"].to(device)
        meta_n = batch["meta_n"].to(device)
        link_pos = batch["link_pos"].to(device)  # (B,6)
        link_neg = batch["link_neg"].to(device)

        z_a = model_enc(loc_a, meta_a)
        z_b = model_enc(loc_b, meta_b)
        z_n = model_enc(loc_n, meta_n)

        # BCE on pos and neg links
        logit_pos = model_cls(z_a, z_b, link_pos)
        logit_neg = model_cls(z_a, z_n, link_neg)

        y_pos = torch.ones_like(logit_pos)
        y_neg = torch.zeros_like(logit_neg)

        loss_bce = bce(logit_pos, y_pos) + bce(logit_neg, y_neg)

        # Triplet on embeddings (continuation closer than non-continuation)
        loss_tri = trip(z_a, z_b, z_n)

        loss = loss_bce + lam * loss_tri

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total += float(loss.detach().cpu().item())

    return total / max(len(loader), 1)


@torch.no_grad()
def eval_one_epoch(model_enc, model_cls, loader, device, lam=0.2, margin=0.2):
    bce = nn.BCEWithLogitsLoss()
    trip = nn.TripletMarginLoss(margin=margin)

    model_enc.eval()
    model_cls.eval()

    total = 0.0
    for batch in loader:
        loc_a = batch["loc_a"].to(device)
        meta_a = batch["meta_a"].to(device)
        loc_b = batch["loc_b"].to(device)
        meta_b = batch["meta_b"].to(device)
        loc_n = batch["loc_n"].to(device)
        meta_n = batch["meta_n"].to(device)
        link_pos = batch["link_pos"].to(device)
        link_neg = batch["link_neg"].to(device)

        z_a = model_enc(loc_a, meta_a)
        z_b = model_enc(loc_b, meta_b)
        z_n = model_enc(loc_n, meta_n)

        logit_pos = model_cls(z_a, z_b, link_pos)
        logit_neg = model_cls(z_a, z_n, link_neg)

        y_pos = torch.ones_like(logit_pos)
        y_neg = torch.zeros_like(logit_neg)

        loss_bce = bce(logit_pos, y_pos) + bce(logit_neg, y_neg)
        loss_tri = trip(z_a, z_b, z_n)
        loss = loss_bce + lam * loss_tri

        total += float(loss.detach().cpu().item())

    return total / max(len(loader), 1)
