import pickle
import torch
from collections import OrderedDict


def original_resnet50_ckpt_loader(mask2former_model, checkpoint_file, strict=False):
    with open(checkpoint_file, "rb") as f:
        original_model = pickle.load(f)

    backbone_weight = OrderedDict()
    pixel_decoder_weight = OrderedDict()
    transformer_decoder_weight = OrderedDict()

    for k, v in original_model["model"].items():
        if "pixel_decoder" in k:
            k: str
            new_k = k.replace("sem_seg_head.pixel_decoder.", "")

            if "adapter_1.norm" in new_k:
                new_k = new_k.replace("adapter_1.norm", "adapter_1.1")
            elif "adapter_1" in new_k:
                new_k = new_k.replace("adapter_1", "adapter_1.0")

            if "layer_1.norm" in new_k:
                new_k = new_k.replace("layer_1.norm", "layer_1.1")
            elif "layer_1" in new_k:
                new_k = new_k.replace("layer_1", "layer_1.0")
            pixel_decoder_weight[new_k] = torch.as_tensor(v)

        elif "predictor" in k:
            new_k = k.replace("sem_seg_head.predictor.", "")

            if "static_query" in new_k:
                new_k = new_k.replace("static_query", "query_feat")

            transformer_decoder_weight[new_k] = torch.as_tensor(v)

        elif "backbone" in k:
            new_k = k.replace("backbone.", "")
            backbone_weight[new_k] = torch.as_tensor(v)
    # %%
    mask2former_model.backbone.load_state_dict(backbone_weight,strict=strict)
    mask2former_model.pixel_decoder.load_state_dict(pixel_decoder_weight, strict=strict)
    mask2former_model.transformer_decoder.load_state_dict(transformer_decoder_weight, strict=strict)
