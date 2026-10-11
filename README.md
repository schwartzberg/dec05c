# CSCI E-89: Problem Set 05

Schwartzberg, Paul

Solutions for the supplied HW5 assignment, developed one problem at a time. The assignment PDF defines the tasks. Some supplied lab guides cover other exercises and are used only where relevant.

## Progress

| Problem | Work | Status |
|---|---|---|
| 1 | Five tiger image transformations using torchvision Compose | Complete; executed and visually checked |
| 2 | BasicCNN with and without batch normalisation, each with and without augmentation | Pending |
| 3 | VGG16 flower classification: frozen-feature stage and fine-tuning stage | Pending |
| 4 | Grad-CAM for the Problem 3 stage 2 flower model | Pending |

The notebook filenames follow the assignment's pattern, beginning with `E89_Schwartzberg_Paul_ProblemSet05_`. Each problem will have an executed notebook and matching HTML export, with explanations beside the code.

## Run Problem 1

Open `E89_Schwartzberg_Paul_ProblemSet05_01.ipynb` and select the `cscie89` kernel. Run its cells in order. The source image is included, so a fresh clone can reproduce the results without downloading it. A missing source image is downloaded automatically.

Alternatively, run `python 01_tiger_transformations.py` from this folder. Results are saved under `outputs/problem1/`.

The existing `cscie89` environment has Python 3.12.14, PyTorch 2.14.0+cpu and torchvision 0.29.0. `requirements.txt` records the package versions used. No additional packages are needed for Problem 1.

## Plan for the flower experiments

Problem 2 follows `1_Augmentation_CNN_Classify_flowers.ipynb` in the Week5 folder. The comparison will use one fixed train/validation/test split and four conditions: the original model and a batch-normalised model, each trained with and without augmentation. Checkpoints will be selected by validation loss; final accuracy and macro precision will be reported over the entire reserved test split. Augmentation will apply only to training images.

Problem 3 follows `3_FineTune_VGG16_PyTorch_Cat_Dog.ipynb`, adapting its binary head and loss to the five flower classes. Stage 1 trains a new classifier with frozen convolutional layers. Stage 2 unfreezes the final convolutional block. Both stages will be evaluated on the full test split, with one illustrated prediction per flower class.

Problem 4 will inspect the stage 2 model with Grad-CAM, following the method in lab 02. The plotted class and target layer will be stated.

Datasets and large weight files are excluded from Git. Code, configuration, split records, metrics, figures, notebooks and HTML reports will be tracked. Every submitted notebook and HTML report will carry the name `Schwartzberg, Paul`.
