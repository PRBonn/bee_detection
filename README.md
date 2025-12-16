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
To reproduce the results of YOLOv5 used in our iterative computer vision pipeline, you will also need to download the model weights:
Test using script from yolov5 repo (see below how to run evaluation script).
+ [iteration 1 checkpoint](https://www.ipb.uni-bonn.de/html/projects/bee_detection_chong2024aeee/checkpoints/round1.pt)
+ [iteration 2 checkpoint](https://www.ipb.uni-bonn.de/html/projects/bee_detection_chong2024aeee/checkpoints/round2.pt)
+ [iteration 3 checkpoint](https://www.ipb.uni-bonn.de/html/projects/bee_detection_chong2024aeee/checkpoints/round3.pt)
+ [iteration 4 checkpoint](https://www.ipb.uni-bonn.de/html/projects/bee_detection_chong2024aeee/checkpoints/round4.pt)
+ [iteration 5 checkpoint](https://www.ipb.uni-bonn.de/html/projects/bee_detection_chong2024aeee/checkpoints/round5.pt)
+ We also have the [weights for YOLOv5 trained on the final dataset](https://www.ipb.uni-bonn.de/html/projects/bee_detection_chong2024aeee/checkpoints/scratch.pt).

### Installation
Our code was tested on a Linux machine, with Python 3.7.13.

Use pip to setup your favourite virtual environment (venv/conda/etc)
```bash
git clone --recurse-submodules https://github.com/PRBonn/bee_detection.git
cd bee_detection
pip install -r requirements.txt
```
You may need to install PyTorch separately if you have specific version requirements.



### Training YOLOv5 for AI-assisted labelling
1. Download the images
2. Download the annotations for training and also the annotations around which to be patched (these are the uncorrected predictions from iteration x.5)
3. Generate the patches 
```bash
python3 crop_BBs.py \
--images_dir <path to image dir> \
--labels_dir <path to labels> \                                  
--patching_labels_dir <path to annotations for location of patches (iteration x.5)> \       
--output_dir <path to output dir> \                        
--num_repeats 1 \
--patch_size 1024
```
4. You may want to visualise the patches as a sanity check (see how to section on visualise)
5. split patches into train-val set. We do not need a test set because we will predict on the full images for AI-assisted annotation and are not interested in the YOLOv5 test performance at this stage.
`python scripts/split_dataset.py <path to parent dir of patches> --test_split 0.0 --val_split 0.15`
5. create a config file pointing to the patches dataset. See yolov5/data/roundx.yaml for an example
5. Train using the patches
```bash
python -m torch.distributed.run --nproc_per_node <number of gpus> --master_port <pick an unused port> train.py --data <config yaml of dataset> --cfg yolov5m.yaml --batch-size 6 --workers 6 --epochs 600 --img 1024 --name <whatever name you would like to use to call this experiment> --project <path to output training logs> --save-period 20 --weights <pretrained weights pt> --cache ram --hyp data/hyps/hyp.scratch-high.yaml;

```

## Generating labels for manual annotation from prediction
```bash
cd scripts;
python merge_yolo_hiwi.py \
--hiwi_labels_dir <dir of labels from previous iteration> \
--round_labels_dir <dir of predictions from current round> \
--out_dir <path to output dir> \
--imgs_dir <path to img dir> 
```


## Evaluating YOLOv5
```bash
python val.py --data <config for test dataset yaml> --weights <weights to evaluate> --batch-size 8 --task "test" --workers 8 --name <experiment name> --img 5120 --project <path to output logs>;
```


## FAQ
Feel free to contact me at linn.chong@uni-bonn.de or open an issue in this repo if you have any questions.
