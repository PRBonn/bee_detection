## Evaluating AI-Assisted Pollinator Detection in Agricultural Fields  
This is the official repository for the paper "Evaluating AI-Assisted Pollinator Detection in Agricultural Fields".

## Paper Highlights
* We introduced an AI-assisted annotation framework for pollinator detection datasets.
* Our annotation framework increased the annotations by 87% compared to manual labels.
* Training with our dataset increased model performance of 29% mAP.
* With our dataset, we examined how different treatments impact model performance.
* We showed that increasing the number of annotated pollinators improves model performance.

## Datasets
Download our [validation data](https://www.ipb.uni-bonn.de/html/projects/bee_detection_chong2024aeee/dataset/final_dataset/bee_chong_seidel_validation_set.zip) as a sample of our dataset.
We also expose the test set [images](https://www.ipb.uni-bonn.de/html/projects/bee_detection_chong2024aeee/dataset/final_dataset/bee_chong_seidel_test_images.zip). 
If you need the annotations for the test set, extract them from the [annotations from each iteration](https://www.ipb.uni-bonn.de/html/projects/bee_detection_chong2024aeee/dataset/iteration_labels.zip).
We will make our full dataset public upon the paper's acceptance. Till then, email me (Linn) for early access to our data.

## Reproducibility
For reproducibility, we also provide the [annotations from each iteration](https://www.ipb.uni-bonn.de/html/projects/bee_detection_chong2024aeee/dataset/iteration_labels.zip).
To fully reproduce the results of YOLOv5 used in our iterative computer vision pipeline, you will also need to download the model weights:
Test using script from yolov5 repo (see below how to run evaluation script).
+ [iteration 1 checkpoint](https://www.ipb.uni-bonn.de/html/projects/bee_detection_chong2024aeee/checkpoints/round1.pt)
+ [iteration 2 checkpoint](https://www.ipb.uni-bonn.de/html/projects/bee_detection_chong2024aeee/checkpoints/round2.pt)
+ [iteration 3 checkpoint](https://www.ipb.uni-bonn.de/html/projects/bee_detection_chong2024aeee/checkpoints/round3.pt)
+ [iteration 4 checkpoint](https://www.ipb.uni-bonn.de/html/projects/bee_detection_chong2024aeee/checkpoints/round4.pt)
+ [iteration 5 checkpoint](https://www.ipb.uni-bonn.de/html/projects/bee_detection_chong2024aeee/checkpoints/round5.pt)
+ We also have the [weights for YOLOv5 trained on the final dataset](https://www.ipb.uni-bonn.de/html/projects/bee_detection_chong2024aeee/checkpoints/scratch.pt).


# Installation
## Requirements
Our code was tested on a Linux machine, with Python 3.7.13.

## Environment setup
Use pip to setup your favourite virtual environment (venv/conda/etc)
```bash
git clone --recurse-submodules https://github.com/PRBonn/bee_detection.git
cd bee_detection
pip install -r requirements.txt
```
You may need to install PyTorch separately if you have specific version requirements.


## Evaluating YOLOv5
```bash
python val.py --data <config for test dataset yaml> --weights <weights to evaluate> --batch-size 8 --task "test" --workers 8 --name <experiment name> --img 5120 --project <path to output logs>;
```


## FAQ
Feel free to contact me at linn.chong@uni-bonn.de or open an issue in this repo if you have any questions.
