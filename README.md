# FaceMaskDetector
Repository is dedicated to face mask detection project using convolutional neural network


## Usage:

1. Training your neural network

```
> python train_mask_detector.py --dataset 'path to the dataset'
```

After the model is trained, add the model weight in ***detect_mask_image.py*** and ***detect_mask_realtime.py***

2. Detecting mask on image

```
> python detect_mask_image.py --image 'path to the png\jpg example'
```

3. Detecting mask in real time

```
> python detect_mask_realtime.py
```
