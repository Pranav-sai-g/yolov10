# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model='yolov8n.pt', source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

def postprocess(self, preds, img, orig_imgs):
    """Post-processes predictions and returns a list of Results objects."""

    # Step 1: Run NMS without specifying classes (if `self.args.classes` didn't work)
    preds = ops.non_max_suppression(
        preds,
        self.args.conf,
        self.args.iou,
        agnostic=self.args.agnostic_nms,
        max_det=self.args.max_det
    )

    if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
        orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

    # Step 2: Define target classes and manually filter detections
    target_classes = [41, 42, 43, 44, 45, 46, 47, 49]  # Replace these IDs with those for apple, orange, etc.
    
    results = []
    for i, pred in enumerate(preds):
        orig_img = orig_imgs[i]

        # Manually filter out predictions not in `target_classes`
        filtered_pred = pred[[cls in target_classes for cls in pred[:, 5].tolist()]]  # Keeps only target class detections

        # Step 3: Process the filtered predictions
        if filtered_pred.shape[0] > 0:  # If there are any detections remaining after filtering
            filtered_pred[:, :4] = ops.scale_boxes(img.shape[2:], filtered_pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=filtered_pred))
        else:
            # Append an empty result if no target class detections were found
            results.append(Results(orig_img, path=self.batch[0][i], names=self.model.names, boxes=[]))

    return results
