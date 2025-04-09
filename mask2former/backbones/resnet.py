from detectron2.config import LazyCall as L, instantiate
from detectron2.modeling.backbone.resnet import ResNet, BasicStem


class DetectronResNet(ResNet):

    def forward(self, x):
        outputs = super().forward(x)
        return list(outputs.values())


def resnet50(pretrained=False):

    resnet_cfg = L(DetectronResNet)(
        stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
        stages=L(ResNet.make_default_stages)(
            depth=50,
            stride_in_1x1=False,
            norm="FrozenBN",
        ),
        out_features=["res2","res3", "res4", "res5"],
        freeze_at=1,
    )

    resnet50 = instantiate(resnet_cfg)

    if pretrained is True:
        import pickle
        import torch
        from collections import OrderedDict

        with open("model_final_3c8ec9.pkl", "rb") as f:
            original_model = pickle.load(f)

        backbone_weight = OrderedDict()

        for k, v in original_model["model"].items():
            if "backbone" in k:
                new_k = k.replace("backbone.", "")
                backbone_weight[new_k] = torch.as_tensor(v)

        resnet50.load_state_dict(backbone_weight)
    return resnet50


def resnet101(pretrained=False):
    pass