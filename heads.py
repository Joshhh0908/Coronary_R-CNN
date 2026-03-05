import torch
import torch.nn as nn
import torch.nn.functional as F

from RPN import encode_boxes_to_deltas, decode_deltas_to_boxes


def decode_deltas_to_boxes_single(anchors_1d: torch.Tensor, deltas_1d: torch.Tensor, Lf: int) -> torch.Tensor:
    """
    Your decode_deltas_to_boxes expects deltas with a batch dimension.
    This wrapper makes it work for [N,2] deltas and returns [N,2].
    """
    if anchors_1d.numel() == 0:
        return anchors_1d
    out = decode_deltas_to_boxes(anchors_1d, deltas_1d.unsqueeze(0), Lf=Lf)  # [1,N,2]
    return out.squeeze(0)

#  Two-bit encoding (paper-style):
# def plaque_label_to_bits(plaque_label: torch.Tensor) -> torch.Tensor:
#     """
#     Two-bit encoding (paper-style):
#       bit0 = calcified present?
#       bit1 = non-calcified present?

#     Labels:
#       0 = background
#       1 = non-calc
#       2 = mixed
#       3 = calcified
#     """
#     y = plaque_label.to(torch.long)
#     bits = torch.zeros((y.numel(), 2), device=y.device, dtype=torch.float32)
#     bits[:, 0] = ((y == 3) | (y == 2)).float()  # calc present
#     bits[:, 1] = ((y == 1) | (y == 2)).float()  # non-calc present
#     return bits



class MLPHead(nn.Module):
    """
    pooled: [N, C, roi_len] -> [N, hidden]
    """
    def __init__(self, in_c: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_c, hidden, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv(pooled), inplace=True)  # [N,hidden,L]
        h = h.mean(dim=-1)                           # [N,hidden]
        h = self.drop(F.relu(self.fc1(h), inplace=True))
        return h


# class PlaqueTwoBitHead(nn.Module):
#     def __init__(self, in_c: int = 512, hidden: int = 256, dropout: float = 0.1):
#         super().__init__()
#         self.backbone = MLPHead(in_c, hidden, dropout)
#         self.out = nn.Linear(hidden, 4)  # logits for (calc_bit, noncalc_bit)

#     def forward(self, pooled: torch.Tensor) -> torch.Tensor:
#         return self.out(self.backbone(pooled))  # [N,2]

class Plaque4ClassHead(nn.Module):
    """
    4-class plaque head.
    Labels:
      0 = background
      1 = non-calc
      2 = mixed
      3 = calcified
    Output: logits [N,4]
    """
    def __init__(self, in_c: int = 512, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.backbone = MLPHead(in_c, hidden, dropout)
        self.out = nn.Linear(hidden, 4)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.out(self.backbone(pooled))  # [N,4]
    
class StenosisClassHead(nn.Module):
    """
    Multi-class stenosis classification head.
    Output: logits [N, K]
    """
    def __init__(self, num_classes: int, in_c: int = 512, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.num_classes = int(num_classes)
        self.backbone = MLPHead(in_c, hidden, dropout)
        self.out = nn.Linear(hidden, self.num_classes)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.out(self.backbone(pooled))  # [N,K]


class RoIRegressionHead(nn.Module):
    """
    Refines proposal boxes via deltas (t_c, t_w).
    """
    def __init__(self, in_c: int = 512, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.backbone = MLPHead(in_c, hidden, dropout)
        self.out = nn.Linear(hidden, 2)  # (t_c, t_w)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.out(self.backbone(pooled))  # [N,2]


class RoIHeads(nn.Module):
    def __init__(self, stenosis_num_classes: int, in_c: int = 512, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.plaque = Plaque4ClassHead(in_c, hidden, dropout)
        self.stenosis = StenosisClassHead(stenosis_num_classes, in_c, hidden, dropout)
        self.roi_reg = RoIRegressionHead(in_c, hidden, dropout)

    def forward(self, pooled: torch.Tensor, rois: torch.Tensor, Lf: int):
        plaque_logits = self.plaque(pooled)             # [R,4]
        stenosis_logits = self.stenosis(pooled)         # [R,K]
        roi_deltas = self.roi_reg(pooled)               # [R,2]
        roi_refined = decode_deltas_to_boxes_single(rois, roi_deltas, Lf=Lf)
        return {
            "plaque_logits": plaque_logits,
            "stenosis_logits": stenosis_logits,
            "roi_deltas": roi_deltas,
            "roi_refined": roi_refined,
        }

    @staticmethod
    def losses(
        out: dict,
        rois: torch.Tensor,
        matched_gt_rois: torch.Tensor,
        matched_plaque: torch.Tensor,
        matched_stenosis: torch.Tensor,
        plaque_w: float = 1.0,
        stenosis_w: float = 1.0,
        roi_reg_w: float = 1.0,
        stenosis_class_weights: torch.Tensor = None,
        plaque_on_pos_only: bool = False,
    ):
        device = out["plaque_logits"].device

        # Plaque (two-bit BCE).  By default we weight only positive
        # examples and optionally limit the loss to positives as well.
        #
        # `pos_weights` is a tensor [2] giving the multiplier for the
        # positive term of each bit; this is appropriate when the issue
        # is missing true plaques rather than being flooded by negatives.
        # If `plaque_on_pos_only` is True we further drop background
        # ROIs from the plaque loss entirely.

        y_plaque = matched_plaque.to(device).long()  # [R]

        is_pos = (matched_plaque.to(device).long() != 0)

        if plaque_on_pos_only:
            if is_pos.any():
                loss_plaque = F.cross_entropy(
                                out["plaque_logits"][is_pos],
                                y_plaque[is_pos])
            else:
                loss_plaque = torch.zeros((), device=device)
        else:
            loss_plaque = F.cross_entropy(
                            out["plaque_logits"], y_plaque)
            
        # Stenosis classification only on positives
        if is_pos.any():
            y = matched_stenosis.to(device).long()
            w = None
            if stenosis_class_weights is not None:
                w = stenosis_class_weights.to(device).float()
            loss_stenosis = F.cross_entropy(out["stenosis_logits"][is_pos], y[is_pos], weight=w)
        else:
            loss_stenosis = torch.zeros((), device=device)
            
        # RoI regression only on positives
        if is_pos.any():
            tgt = encode_boxes_to_deltas(rois[is_pos], matched_gt_rois[is_pos])  # [P,2]
            loss_roi = F.smooth_l1_loss(out["roi_deltas"][is_pos], tgt)
        else:
            loss_roi = torch.zeros((), device=device)

        total = plaque_w * loss_plaque + stenosis_w * loss_stenosis + roi_reg_w * loss_roi

        stats = {
            "loss_total": float(total.detach().cpu()),
            "loss_plaque": float(loss_plaque.detach().cpu()),
            "loss_stenosis": float(loss_stenosis.detach().cpu()),
            "loss_roi_reg": float(loss_roi.detach().cpu()),
            "num_pos_rois": int(is_pos.sum().item()),
        }
        return total, stats
