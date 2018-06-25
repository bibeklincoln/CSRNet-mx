# CSRNet
This is the **MXNet** implementation for [CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes](https://arxiv.org/abs/1802.10062) in CVPR 2018.
And we also provide the **Caffe** implementation in this repo. :-)

[[Official Repo](https://github.com/leeyeehoo/CSRNet)]

## Datasets
ShanghaiTech Dataset: [Google Drive](https://drive.google.com/open?id=16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI)

## Models (Only for tests)

This is the model for test. The results should be similar to the results shown in the paper(slightly better or worse).

1) ShanghaiTech_Part_A: [Google Drive](https://drive.google.com/open?id=1odZ3B_ZDSepPcVFO_TfGUIrpF2DF7SwY)

2) ShanghaiTech_Part_B: [Google Drive](https://drive.google.com/open?id=1NOpn0ztlye85vrHR2TMwOI2Qu_S8zANj)

## Setup
```bash
git clone https://github.com/wkcn/CSRNet-mx
git submodule update --init --recursive
```

## Convert Caffe Model to MXNet Model (If using MXNet)
Please install pycaffe for converting model
Download the model to *models* directory.
```
python caffe2mx.py
```

## Dataset
```
data/
    ShanghaiTech/
        part_A_final/
        part_B_final/
```

## Prediction 
The prediction result will be saved in the file *predict.txt*.
- MXNet
```
python test.py
```
- Caffe 
```
python test_caffe.py
```

## Evaluation
```
python evaluate.py
```

## References

```
@article{li2018csrnet,
  title={CSRNet: Dilated convolutional neural networks for understanding the highly congested scenes},
  author={Li, Yuhong and Zhang, Xiaofan and Chen, Deming},
  journal={arXiv preprint arXiv:1802.10062},
  year={2018}
}
```
