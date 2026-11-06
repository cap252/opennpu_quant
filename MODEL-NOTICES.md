# Model notices

This document maps the 30 supported models to their upstream sources, license
texts and conversion steps. The Python package's Apache-2.0 license does not
replace the terms of the model weights distributed in separate Release archives.
Recording an attribution does not by itself establish redistribution permission.

Model links below open the recipes, which contain the checkpoint URL and SHA256.
The source-license column describes the referenced source tree. Weight terms are
listed separately. "Not established" means that checkpoint-specific terms have
not been verified here, not that the weights are public domain or prohibited.

Recipe `source.license` values summarize upstream code notices. They do not
establish checkpoint-specific redistribution terms. This record collects the
available evidence and does not replace the upstream license terms.

## Models and conversions

All models are exported to FP32 ONNX. The export column lists additional
model-specific processing. RTN variants add INT8 QuantizeLinear/DequantizeLinear
nodes and quantized parameters. AdaRound variants optimize weight rounding using
calibration inputs before applying those quantized parameters. The Release
catalog identifies the candidate, quantization settings and resulting graph
hashes. Recipes describe the export configuration, not a complete build log for
every historical asset.

| Model and recipe | Source license | Weight-license evidence | Export treatment |
|---|---|---|---|
| [DeepLabV3-MobileNetV2](src/opennpu_quant/resources/models/deeplabv3_mnv2_voc.yaml) | [TensorFlow DeepLab][tensorflow/deeplab] · Apache-2.0 at the model publication revision | Checkpoint-specific terms not established | TensorFlow graph extraction, logits restoration in a host graph |
| [DeepLabV3-ResNet50](src/opennpu_quant/resources/models/deeplabv3_r50_voc.yaml) | [MMSegmentation][open-mmlab/mmsegmentation] · Apache-2.0 | Checkpoint-specific terms not established | MMSeg backbone and decode head, fixed 512×512 input |
| [EfficientNet-B0](src/opennpu_quant/resources/models/efficientnet_b0.yaml) | [torchvision][pytorch/vision] · BSD-3-Clause | Checkpoint-specific terms not established | FP32 network export |
| [EfficientNet-Lite4](src/opennpu_quant/resources/models/efficientnet_lite4.yaml) | [timm][huggingface/pytorch-image-models] · Apache-2.0 | Apache-2.0 in the pinned model card | timm export with fixed-size SAME padding |
| [FCN-ResNet50](src/opennpu_quant/resources/models/fcn_r50_voc.yaml) | [MMSegmentation][open-mmlab/mmsegmentation] · Apache-2.0 | Checkpoint-specific terms not established | MMSeg backbone and decode head, fixed 512×512 input |
| [Inception-V3](src/opennpu_quant/resources/models/inception_v3.yaml) | [torchvision][pytorch/vision] · BSD-3-Clause | Checkpoint-specific terms not established | Input transform in a host graph |
| [MobileFaceNet ReLU (foamliu)](src/opennpu_quant/resources/models/mobilefacenet_relu_foamliu.yaml) | [MobileFaceNet][foamliu/MobileFaceNet] · Apache-2.0 | Checkpoint included in the Apache-2.0 source tree. No separate weight notice | Original ReLU network, no PReLU-to-ReLU replacement |
| [MobileNetV1](src/opennpu_quant/resources/models/mobilenet_v1.yaml) | [timm][huggingface/pytorch-image-models] · Apache-2.0 | Apache-2.0 in the pinned model card | timm export with fixed-size SAME padding |
| [MobileNetV2](src/opennpu_quant/resources/models/mobilenet_v2.yaml) | [torchvision][pytorch/vision] · BSD-3-Clause | Checkpoint-specific terms not established | FP32 network export |
| [MobileNetV3-Large](src/opennpu_quant/resources/models/mobilenet_v3_large.yaml) | [torchvision][pytorch/vision] · BSD-3-Clause | Checkpoint-specific terms not established | FP32 network export |
| [NanoDet-Plus](src/opennpu_quant/resources/models/nanodet_plus_m_320.yaml) | [NanoDet][RangiLyu/nanodet] · Apache-2.0 | Checkpoint-specific terms not established | Checkpoint key mapping, ONNX head, separate postprocess graph |
| [ResNet-101](src/opennpu_quant/resources/models/resnet101.yaml) | [torchvision][pytorch/vision] · BSD-3-Clause | Checkpoint-specific terms not established | FP32 network export |
| [ResNet-18](src/opennpu_quant/resources/models/resnet18.yaml) | [torchvision][pytorch/vision] · BSD-3-Clause | Checkpoint-specific terms not established | FP32 network export |
| [ResNet-50](src/opennpu_quant/resources/models/resnet50.yaml) | [torchvision][pytorch/vision] · BSD-3-Clause | Checkpoint-specific terms not established | FP32 network export |
| [RetinaFace MobileNet0.25 (640x640)](src/opennpu_quant/resources/models/retinaface_mnet025_640.yaml) | [RetinaFace][biubug6/Pytorch_Retinaface] · MIT | Checkpoint-specific terms not established | Network and postprocess graphs |
| [RetinaNet](src/opennpu_quant/resources/models/retinanet_resnet50_fpn.yaml) | [torchvision][pytorch/vision] · BSD-3-Clause | Checkpoint-specific terms not established | Input transform, network and postprocess graphs |
| [ShuffleNetV2 1.0x](src/opennpu_quant/resources/models/shufflenet_v2_x1_0.yaml) | [torchvision][pytorch/vision] · BSD-3-Clause | Checkpoint-specific terms not established | FP32 network export |
| [SqueezeNet 1.1](src/opennpu_quant/resources/models/squeezenet1_1.yaml) | [torchvision][pytorch/vision] · BSD-3-Clause | Checkpoint-specific terms not established | FP32 network export |
| [SSD-MobileNetV2](src/opennpu_quant/resources/models/ssd_mobilenet_v2_320.yaml) | [TensorFlow Models][tensorflow/models] · Apache-2.0 | Checkpoint-specific terms not established | TensorFlow conversion, CPU input/output graphs around the network |
| [U-Net-ResNet50](src/opennpu_quant/resources/models/unet_resnet50_voc.yaml) | [unet-pytorch][bubbliiiing/unet-pytorch] · MIT | Checkpoint-specific terms not established | FP32 network export |
| [VGG-16](src/opennpu_quant/resources/models/vgg16.yaml) | [torchvision][pytorch/vision] · BSD-3-Clause | Checkpoint-specific terms not established | FP32 network export |
| [YOLOv3-Tiny](src/opennpu_quant/resources/models/yolov3_tiny.yaml) | [YOLOv3][ultralytics/yolov3] · GPL-3.0 | Historical GPL source. Checkpoint scope still needs confirmation | Network and postprocess graphs |
| [YOLOv4-Tiny](src/opennpu_quant/resources/models/yolov4_tiny.yaml) | [pytorch-YOLOv4][Tianxiaomo/pytorch-YOLOv4] · Apache-2.0 | AlexeyAB/Darknet weights. Converter license does not establish weight terms | Darknet weights loaded in a PyTorch implementation, separate postprocess graph |
| [YOLOv5n](src/opennpu_quant/resources/models/yolov5n.yaml) | [YOLOv5][ultralytics/yolov5] · GPL-3.0 | Historical GPL source. Checkpoint scope still needs confirmation | Network and postprocess graphs |
| [YOLOv5s](src/opennpu_quant/resources/models/yolov5s.yaml) | [YOLOv5][ultralytics/yolov5] · GPL-3.0 | Historical GPL source. Checkpoint scope still needs confirmation | Network and postprocess graphs |
| [YOLOv5s-seg](src/opennpu_quant/resources/models/yolov5s_seg.yaml) | [YOLOv5][ultralytics/yolov5] · GPL-3.0 | Historical GPL source. Checkpoint scope still needs confirmation | Detection and prototype outputs, separate detection postprocess graph |
| [YOLOv7-face](src/opennpu_quant/resources/models/yolov7_face.yaml) | [YOLOv7-face][lucasjinreal/yolov7-face] · Darknet notice only | Mixed upstream provenance. Weight terms not established | Fixed 640×640 input, constant/broadcast rewrites, separate postprocess graph |
| [YOLOv7-tiny](src/opennpu_quant/resources/models/yolov7_tiny.yaml) | [YOLOv7][WongKinYiu/yolov7] · GPL-3.0 notice in a subsequent revision with identical executable code, as detailed below | Checkpoint applicability and source-distribution obligations need confirmation | Network and postprocess graphs |
| [YOLOv8n](src/opennpu_quant/resources/models/yolov8n.yaml) | [Ultralytics][ultralytics/ultralytics] · AGPL-3.0 | Provider offers AGPL-3.0 and separate Enterprise terms. No Enterprise grant recorded | Network and postprocess graphs |
| [YOLOX](src/opennpu_quant/resources/models/yolox.yaml) | [YOLOX][Megvii-BaseDetection/YOLOX] · Apache-2.0 | Checkpoint-specific terms not established | Decoded network outputs, separate postprocess graph |

## Original notices

The following copies retain the upstream text, including copyright and
attribution lines. Identical texts share one file, with each source listed below.
The existing MobileFaceNet copy also has a source URL header. Source revisions
are recorded here so a later license change is not silently applied to an older
checkpoint.

| Source | Revision examined | Upstream original | Local copy |
|---|---|---|---|
| [AlexeyAB/darknet][AlexeyAB/darknet] | `9d40b619756be9521bc2ccd81808f502daaa3e9a` | [LICENSE](https://raw.githubusercontent.com/AlexeyAB/darknet/9d40b619756be9521bc2ccd81808f502daaa3e9a/LICENSE) | [Text](licenses/models/darknet.LICENSE) |
| [biubug6/Pytorch_Retinaface][biubug6/Pytorch_Retinaface] | `b984b4b775b2c4dced95c1eadd195a5c7d32a60b` | [LICENSE.MIT](https://raw.githubusercontent.com/biubug6/Pytorch_Retinaface/b984b4b775b2c4dced95c1eadd195a5c7d32a60b/LICENSE.MIT) | [Text](licenses/models/retinaface.LICENSE.MIT) |
| [bubbliiiing/unet-pytorch][bubbliiiing/unet-pytorch] | `5bcd6b4c832648beed1b92e78ed1e85c56343eca` | [LICENSE](https://raw.githubusercontent.com/bubbliiiing/unet-pytorch/5bcd6b4c832648beed1b92e78ed1e85c56343eca/LICENSE) | [Text](licenses/models/unet-pytorch.LICENSE) |
| [foamliu/MobileFaceNet][foamliu/MobileFaceNet] | `a6cc9032a659b615f477833e1a70b5e7931bcccc` | [LICENSE](https://raw.githubusercontent.com/foamliu/MobileFaceNet/a6cc9032a659b615f477833e1a70b5e7931bcccc/LICENSE) | [Text](src/opennpu_quant/resources/licenses/mobilefacenet.txt) |
| [huggingface/pytorch-image-models][huggingface/pytorch-image-models] | `e44f14d7d2f557b9f3add82ee4f1ed2beefbb30d` | [LICENSE](https://raw.githubusercontent.com/huggingface/pytorch-image-models/e44f14d7d2f557b9f3add82ee4f1ed2beefbb30d/LICENSE) | [Text](licenses/models/timm.LICENSE) |
| [lucasjinreal/yolov7-face][lucasjinreal/yolov7-face] | `adeb529a1189ad253dabb4461f8184745f07c4e8` | [LICENSE.md](https://raw.githubusercontent.com/lucasjinreal/yolov7-face/adeb529a1189ad253dabb4461f8184745f07c4e8/LICENSE.md) | [Text](licenses/models/darknet.LICENSE) |
| [Megvii-BaseDetection/YOLOX][Megvii-BaseDetection/YOLOX] | `6ddff4824372906469a7fae2dc3206c7aa4bbaee` | [LICENSE](https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/6ddff4824372906469a7fae2dc3206c7aa4bbaee/LICENSE) | [Text](licenses/models/yolox.LICENSE) |
| [open-mmlab/mmsegmentation][open-mmlab/mmsegmentation] | `b040e147adfa027bbc071b624bedf0ae84dfc922` | [LICENSE](https://raw.githubusercontent.com/open-mmlab/mmsegmentation/b040e147adfa027bbc071b624bedf0ae84dfc922/LICENSE) | [Text](licenses/models/mmsegmentation.LICENSE) |
| [pytorch/vision][pytorch/vision] | `3ac97aa9120137381ed1060f37237e44485ac2aa` | [LICENSE](https://raw.githubusercontent.com/pytorch/vision/3ac97aa9120137381ed1060f37237e44485ac2aa/LICENSE) | [Text](licenses/models/torchvision.LICENSE) |
| [RangiLyu/nanodet][RangiLyu/nanodet] | `d3fb34fa91d6020f273d6d063bf324dcd97bac12` | [LICENSE](https://raw.githubusercontent.com/RangiLyu/nanodet/d3fb34fa91d6020f273d6d063bf324dcd97bac12/LICENSE) | [Text](licenses/models/nanodet.LICENSE) |
| [TensorFlow DeepLab][tensorflow/deeplab] | `7a2f1a3e09b7a3f8f987791a740a2da8bdd226c0`, model publication revision | [LICENSE](https://raw.githubusercontent.com/tensorflow/models/7a2f1a3e09b7a3f8f987791a740a2da8bdd226c0/LICENSE) | [Text](licenses/models/tensorflow-deeplab-2018.LICENSE) |
| [tensorflow/models][tensorflow/models] | `8b12ae202a3ccf8f965c730a4e7617204e32000b` | [LICENSE](https://raw.githubusercontent.com/tensorflow/models/8b12ae202a3ccf8f965c730a4e7617204e32000b/LICENSE) | [Text](licenses/models/tensorflow-models.LICENSE) |
| [Tianxiaomo/pytorch-YOLOv4][Tianxiaomo/pytorch-YOLOv4] | `a65d219f9066bae4e12003bd7cdc04531860c672` | [License.txt](https://raw.githubusercontent.com/Tianxiaomo/pytorch-YOLOv4/a65d219f9066bae4e12003bd7cdc04531860c672/License.txt) | [Text](licenses/models/pytorch-yolov4.License.txt) |
| [ultralytics/ultralytics][ultralytics/ultralytics] | `6e43d1e1e5db72afbf686dee6745669bcb124b0a` | [LICENSE](https://raw.githubusercontent.com/ultralytics/ultralytics/6e43d1e1e5db72afbf686dee6745669bcb124b0a/LICENSE) | [Text](licenses/models/ultralytics.LICENSE) |
| [ultralytics/yolov3][ultralytics/yolov3] | `7eb23e3c1d387cdcefbad266046af369a9f40399` | [LICENSE](https://raw.githubusercontent.com/ultralytics/yolov3/7eb23e3c1d387cdcefbad266046af369a9f40399/LICENSE) | [Text](licenses/models/yolov3-yolov5.LICENSE) |
| [ultralytics/yolov5][ultralytics/yolov5] | `915bbf294bb74c859f0b41f1c23bc395014ea679` | [LICENSE](https://raw.githubusercontent.com/ultralytics/yolov5/915bbf294bb74c859f0b41f1c23bc395014ea679/LICENSE) | [Text](licenses/models/yolov3-yolov5.LICENSE) |
| [WongKinYiu/yolov7][WongKinYiu/yolov7] | `8fb51236492095eb55ea426ffdeb943f46f17289`, subsequent notice revision | [LICENSE.md](https://raw.githubusercontent.com/WongKinYiu/yolov7/8fb51236492095eb55ea426ffdeb943f46f17289/LICENSE.md) | [Text](licenses/models/yolov7.LICENSE.md) |

The YOLOv4-Tiny configuration is pinned to Darknet revision `59596d7`, whose
LICENSE is identical to the retained text from `9d40b61`.

The pinned YOLOv7 revision `9a00c17` has no license file. The subsequent
`8fb5123` revision added GPL-3.0 on the same day. The [revision comparison](https://github.com/WongKinYiu/yolov7/compare/9a00c173a44bfcf3a58cc2a60e37deb6a1fb9e03...8fb51236492095eb55ea426ffdeb943f46f17289)
contains only that addition and a README edit, so the executable source is
identical. This is source-license evidence, not confirmation of the checkpoint's
redistribution terms or fulfillment of any corresponding-source obligations.

## Weight terms requiring care

- **Torchvision:** its [pretrained-model notice][torchvision-weights] says model
  terms can differ from the library license. The BSD source notice is not a
  checkpoint-specific license grant. Restrictions on other Torchvision models
  are not assigned to these checkpoints.
- **MobileNetV1 and EfficientNet-Lite4:** the pinned [MobileNetV1 model
  card][mobilenet-card] and [EfficientNet-Lite4 model card][efficientnet-card]
  explicitly identify Apache-2.0 for the weights.
- **MobileFaceNet:** the checkpoint and Apache-2.0 notice are in the same pinned
  repository. Applying that notice to the checkpoint is an inference from that
  distribution, not a separate model-card statement.
- **YOLOv3, YOLOv5 and YOLOv7-tiny:** the referenced YOLOv3 and YOLOv5 trees
  contain GPL-3.0. For YOLOv7-tiny, a subsequent revision added the GPL-3.0
  notice with unchanged executable source, as described above. Checkpoint
  applicability and corresponding-source obligations for converted assets
  still need confirmation.
- **YOLOv8:** the pinned [provider documentation][ultralytics-terms] describes
  AGPL-3.0 and separately agreed Enterprise licensing. No Enterprise agreement
  is asserted here. Applicable source-distribution obligations are not fulfilled
  merely by copying a license text.
- **YOLOv4 and YOLOv7-face:** YOLOv4 uses an Apache-2.0 converter with weights from
  a different provider. The Darknet notice copied from that provider also occurs
  in the YOLOv7-face fork. It does not establish the terms of every component or
  checkpoint in either distribution.
- **DeepLabV3-MobileNetV2:** the [official Model Zoo][deeplab-model-zoo] publishes
  `deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz`, whose SHA256 matches the
  recipe. The cited `7a2f1a3` revision records its publication, not the code used
  to train the checkpoint. Export reads the published frozen graph. The source
  notice is Apache-2.0. Checkpoint-specific redistribution terms remain unverified.

The remaining rows marked "Checkpoint-specific terms not established" need
weight-specific confirmation. Model and dataset terms are separate. Datasets
are not included in the model archives.

## Fixed-input side edition

`v1.0.2-static.1` fixes the network input dimensions of YOLOX, YOLOv7-face and
YOLOv5s-seg to 640×640, and DeepLabV3-ResNet50 and FCN-ResNet50 to 512×512.
Converted Release graphs preserve nodes, weights and quantization encodings. The
latter four models use the resize and padding policies in [the model
reference](https://github.com/cap252/opennpu_quant/blob/v1.0.2-static.1/docs/model-reference.md).
This conversion does not change the upstream notices or establish additional
weight redistribution rights.

## Maintaining this record

When adding or replacing a checkpoint, update its recipe and model row, retain
its upstream notice unchanged and record the source revision. Keep unresolved
weight terms explicit. The model table and original texts are documentation,
not an execution-time registry.

The v1.0.2-static.1 model archives include this record and the original
notices under `licenses/models/`.

[AlexeyAB/darknet]: https://github.com/AlexeyAB/darknet/tree/9d40b619756be9521bc2ccd81808f502daaa3e9a
[biubug6/Pytorch_Retinaface]: https://github.com/biubug6/Pytorch_Retinaface/tree/b984b4b775b2c4dced95c1eadd195a5c7d32a60b
[bubbliiiing/unet-pytorch]: https://github.com/bubbliiiing/unet-pytorch/tree/5bcd6b4c832648beed1b92e78ed1e85c56343eca
[foamliu/MobileFaceNet]: https://github.com/foamliu/MobileFaceNet/tree/a6cc9032a659b615f477833e1a70b5e7931bcccc
[huggingface/pytorch-image-models]: https://github.com/huggingface/pytorch-image-models/tree/e44f14d7d2f557b9f3add82ee4f1ed2beefbb30d
[lucasjinreal/yolov7-face]: https://github.com/lucasjinreal/yolov7-face/tree/adeb529a1189ad253dabb4461f8184745f07c4e8
[Megvii-BaseDetection/YOLOX]: https://github.com/Megvii-BaseDetection/YOLOX/tree/6ddff4824372906469a7fae2dc3206c7aa4bbaee
[open-mmlab/mmsegmentation]: https://github.com/open-mmlab/mmsegmentation/tree/b040e147adfa027bbc071b624bedf0ae84dfc922
[pytorch/vision]: https://github.com/pytorch/vision/tree/3ac97aa9120137381ed1060f37237e44485ac2aa
[RangiLyu/nanodet]: https://github.com/RangiLyu/nanodet/tree/d3fb34fa91d6020f273d6d063bf324dcd97bac12
[tensorflow/deeplab]: https://github.com/tensorflow/models/tree/7a2f1a3e09b7a3f8f987791a740a2da8bdd226c0/research/deeplab
[deeplab-model-zoo]: https://github.com/tensorflow/models/blob/7a2f1a3e09b7a3f8f987791a740a2da8bdd226c0/research/deeplab/g3doc/model_zoo.md
[tensorflow/models]: https://github.com/tensorflow/models/tree/8b12ae202a3ccf8f965c730a4e7617204e32000b
[Tianxiaomo/pytorch-YOLOv4]: https://github.com/Tianxiaomo/pytorch-YOLOv4/tree/a65d219f9066bae4e12003bd7cdc04531860c672
[ultralytics/ultralytics]: https://github.com/ultralytics/ultralytics/tree/6e43d1e1e5db72afbf686dee6745669bcb124b0a
[ultralytics/yolov3]: https://github.com/ultralytics/yolov3/tree/7eb23e3c1d387cdcefbad266046af369a9f40399
[ultralytics/yolov5]: https://github.com/ultralytics/yolov5/tree/915bbf294bb74c859f0b41f1c23bc395014ea679
[WongKinYiu/yolov7]: https://github.com/WongKinYiu/yolov7/tree/9a00c173a44bfcf3a58cc2a60e37deb6a1fb9e03
[torchvision-weights]: https://github.com/pytorch/vision/blob/3ac97aa9120137381ed1060f37237e44485ac2aa/README.md#pre-trained-model-license
[mobilenet-card]: https://huggingface.co/timm/mobilenetv1_100.ra4_e3600_r224_in1k/raw/6e88b523abded44469eafcd52f7832d5e0300b1b/README.md
[efficientnet-card]: https://huggingface.co/timm/tf_efficientnet_lite4.in1k/raw/9006f1cfa1c536cdfc0a2d7011d075179e69a9ca/README.md
[ultralytics-terms]: https://github.com/ultralytics/ultralytics/blob/6e43d1e1e5db72afbf686dee6745669bcb124b0a/README.md#license
