# zindi-arm-unicef-dvc

## Zindi Arm UNICEF Disaster Vulnerability Challenge

This is my entry for this competition:
https://zindi.africa/competitions/arm-unicef-disaster-vulnerability-challenge

This is a fairly straightforward object detection and classification task,
here using YOLOv8. That already does a lot of training data augmentation
for us.

I have explored the augmentation options and some key hyperparameters to try
and get optimal results. Of particular importance are:

* conf: Confidence threshold; trade-off between false negatives and false
        positives.
* iou: Intersection over Union threshold for Non-Maximal Suppression, i.e.
    rejecting lower-confidence detections which overlap a stronger detection
    to some degree. As houses don't overlap (unlike some other probems where
    we may have a dog behind a cat say), this is set fairly low.
    