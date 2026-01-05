import torch
from torch.utils.data import DataLoader

from dataprep import VesselWindowDataset, collate_fn
from detection_module import FeatureExtractorFE, RPN1D, rpn_loss_1d


def main():
    csv_path = "/home/joshua/Coronary_R-CNN/train_cpr_all26_allbranch_02mm.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ds = VesselWindowDataset(csv_path=csv_path, window_len=768, train=True, do_augment=True)
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn)

    fe = FeatureExtractorFE(in_ch=1, norm="gn").to(device)
    rpn = RPN1D(in_c=512, anchor_lengths=(2,4,6,9,13,18)).to(device)

    params = list(fe.parameters()) + list(rpn.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)

    fe.train()
    rpn.train()

    for step, (x, targets) in enumerate(dl):
        x = x.to(device, non_blocking=True)

        feat = fe(x)                      # [B,512,48]
        obj_logits, deltas, anchors = rpn(feat)   # [B,288], [B,288,2], [288,2]

        loss, stats = rpn_loss_1d(
            obj_logits, deltas, anchors, targets,
            stride=16.0,
            pos_iou_thresh=0.5,
            neg_iou_thresh=0.1,
            sample_size=256,
            pos_fraction=0.5,
            reg_weight=1.0,
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

        if step % 10 == 0:
            print(f"step {step:04d} | total {stats['loss_total']:.4f} "
                  f"obj {stats['loss_obj']:.4f} reg {stats['loss_reg']:.4f} "
                  f"pos {stats['pos_anchors']} samp {stats['sampled_anchors']}")

        if step >= 200:
            break


if __name__ == "__main__":
    main()
