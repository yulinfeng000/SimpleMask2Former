import torch
import torch.nn as nn
import torch.nn.functional as F


class Mask2Former(nn.Module):

    def __init__(self, backbone, neck, pixel_decoder, transformer_decoder):
        super().__init__()
        self.backbone = backbone
        self.neck = neck if neck is not None else nn.Identity()
        self.pixel_decoder = pixel_decoder
        self.transformer_decoder = transformer_decoder

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        mask_feature, multi_scale_memories = self.pixel_decoder(x)
        outputs = self.transformer_decoder(multi_scale_memories, mask_feature)
        return outputs


@torch.no_grad()
def postprocessing_instance_segmentation(
    pred_logits, pred_masks, metainfos, threshold=0.5
):
    """
    Args:
        pred_logits: N,num_queries,N_classes
        pred_masks:  N,num_queries,H,W
        metainfos:   List[Dict[str,...]]
    """
    device = pred_logits.device
    num_classes = pred_logits.shape[-1] - 1
    num_queries = pred_logits.shape[-2]
    img_sizes   = [metafino["img_shape"] for metafino in metainfos]

    pred_results = []

    for mask_pred, mask_cls, img_size in zip(pred_masks, pred_logits, img_sizes):
        pred_mask_target_size = img_size

        mask_pred = F.interpolate(
            mask_pred.unsqueeze(0),
            size=(pred_mask_target_size[0], 
                  pred_mask_target_size[1]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        scores = torch.nn.functional.softmax(mask_cls, dim=-1)[:, :-1]

        labels = (
            torch.arange(num_classes, device=device)
            .unsqueeze(0)
            .repeat(num_queries, 1)
            .flatten(0, 1)
        )

        scores_per_image, topk_indices = scores.flatten(0, 1).topk(
            num_queries, sorted=False
        )

        labels_per_image = labels[topk_indices]

        topk_indices = torch.div(topk_indices, num_classes, rounding_mode="floor")
        mask_pred = mask_pred[topk_indices]
        mask_pred = (mask_pred > 0).float()

        # Calculate average mask prob
        mask_scores_per_image = (
            mask_pred.sigmoid().flatten(1) * mask_pred.flatten(1)
        ).sum(1) / (mask_pred.flatten(1).sum(1) + 1e-6)

        pred_scores = scores_per_image * mask_scores_per_image
        pred_classes = labels_per_image

        masks_results = mask_pred[pred_scores > threshold].bool()
        labels_results = pred_classes[pred_scores > threshold].long()
        scores_results = pred_scores[pred_scores > threshold].float()

        pred_results.append(
            dict(masks=masks_results, labels=labels_results, scores=scores_results)
        )

    return pred_results
