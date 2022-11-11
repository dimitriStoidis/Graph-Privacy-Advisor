# Content-based Graph-Privacy-Advisor
Graph Privacy Advisor (GPA) is an image privacy classifier that uses scene and object cardinality information to predict the privacy of an image.
GPA refines the relevance of the information extracted from the image and determines the most informative features to be used for the privacy classification task.
![Graph Privacy Advisor pipeline](/GPA_pipeline.png)

# Installation

## Requirements
* opencv-python==4.5.5.62
* torchvision=0.9.1
* python=3.9.6
* pytorch=1.8.1
* numpy=1.20.3

## Download datasets
* PicAlert: http://l3s.de/picalert/
* VISPR: https://tribhuvanesh.github.io/vpa/
* PrivacyAlert: https://zenodo.org/record/6406870#.Y2KtsdLP3ow

## Instructions
1. Clone repository</br>
`git clone https://github.com/dimitriStoidis/Graph-Privacy-Advisor`

2. From a terminal or an Anaconda Prompt, go to project's root directory
and run:</br>
`conda create gpa` </br>
`conda activate gpa` </br>
and install the required packages


## Training example

To train the model run: </br>
`python main.py --model_name model1 --num_epochs 50 --batch_size 64 --cardinality True --scene True`


## Evaluation

### References
The work is based on:
* https://github.com/guang-yanng/Image_Privacy
* https://github.com/HCPLab-SYSU/SR
* https://github.com/CSAILVision/places365
* https://pjreddie.com/darknet/yolo/

### Cite
```
@misc{https://doi.org/10.48550/arxiv.2210.11169,
  doi = {10.48550/ARXIV.2210.11169},
  url = {https://arxiv.org/abs/2210.11169},
  author = {Stoidis, Dimitrios and Cavallaro, Andrea},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Content-based Graph Privacy Advisor},
  publisher = {IEEE BigMM},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
### Contact
For any enquiries contact dimitrios.stoidis@qmul.ac.uk, a.cavallaro@qmul.ac.uk

### Licence
This work is licensed under the [MIT License](https://github.com/dimitriStoidis/GenGAN/blob/main/LICENSE).
