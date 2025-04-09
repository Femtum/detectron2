## How to use torchscript_inference.py

1. Using terminal, `cd` in demo directory (`detectron2/demo`)
2. Export model to Torchscript
   ```bash
   python ../../tools/deploy/export_model.py --config-file ../../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
       --output ./output --export-method scripting --format torchscript \
       MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl \
       MODEL.DEVICE cpu
   ```
3. Execute `torchscript_inference.py`
