# CSCI E-89: Problem Set 05

Schwartzberg, Paul

All four problems are complete. The combined notebook was run from the beginning in one fresh `cscie89` kernel. All 28 code cells passed, and its HTML export contains the completed run. The assignment PDF defines the tasks; the supplied CIFAR-10 homework guide covers a different exercise.

## Submission files

The complete submission is one notebook and its matching HTML export:

- `E89_Schwartzberg_Paul_ProblemSet05_01.ipynb`
- `E89_Schwartzberg_Paul_ProblemSet05_01.html`

All four problems appear in this notebook, with a contents list, numbered sections, code, results and explanations. Shared data helpers are defined once in Problem 2; the VGG16 architecture is defined in Problem 3 and reused for Grad-CAM in Problem 4. The notebook contains its solution definitions and requires no local helper-module imports.

The earlier separate notebooks and their HTML reports are kept in `individual_problems/` as working records. They are not additional submission files.

## Measured results

All flower runs use the same stratified split: 2,569 training, 550 validation and 551 test images, with seed 42. Exact image paths are saved in `outputs/shared/split_manifest.json`. Checkpoint selection uses validation loss. Every reported test score uses all 551 test images.

| Problem 2 condition | Selected epoch | Test accuracy | Test macro precision |
|---|---:|---:|---:|
| original_no_augmentation | 11 | 64.61% | 64.35% |
| batchnorm_no_augmentation | 9 | 70.96% | 70.98% |
| original_with_augmentation | 24 | 68.60% | 68.99% |
| batchnorm_with_augmentation | 23 | 74.77% | 74.38% |

| Problem 3 stage | Selected epoch | Test accuracy | Test macro precision |
|---|---:|---:|---:|
| stage1 | 5 | 87.11% | 87.49% |
| stage2 | 3 | 89.29% | 89.87% |

Both VGG16 stages recognised the same five fixed gallery examples. The complete test report also records errors and a decline in rose recall after fine-tuning. Problem 4 correctly predicts its fixed validation daisy; Grad-CAM highlights part of its upper-right petals, while the yellow centre is weak. The notebook explains that limitation.

## Run everything yourself

In Anaconda Prompt:

```bat
conda activate cscie89
cd /d "C:\Users\schwa\Documents\H_DCE\CSCI E-89 Deep Learning\Week5\Assignment\dec05c"
python run_all.py
```

This executes every cell of the combined notebook in one fresh kernel, trains the four CNN conditions and both VGG16 stages again, saves the executed notebook and its HTML report, and runs `check_submission.py`. It does not replace training with cached metrics. Allow roughly half an hour on this CPU; runtime depends on other work on the computer. The recorded combined run took 31.1 minutes, including all training and notebook outputs.

Save and close notebooks before running this command. The runner backs up the notebook in the Windows temporary folder and checks for concurrent edits before replacing outputs. A failed run retains the original and saves a diagnostic copy. To run an earlier separate working notebook, use `python run_all.py --problems 2` (or another number). This optional command uses `individual_problems/`; the default command runs the combined submission. Problem 4 needs the checkpoint created by Problem 3.

To run interactively, start `jupyter notebook` from this folder, select the `cscie89` kernel, and open `E89_Schwartzberg_Paul_ProblemSet05_01.ipynb`, restart its kernel, and use **Run All Cells**. Run the whole notebook from top to bottom; later sections reuse variable names. The HTML shows the completed run; your rerun will update it.

The existing environment has Python 3.12.14, PyTorch 2.14.0+cpu and torchvision 0.29.0. `requirements.txt` records packages and `outputs/environment.json` records the actual environment. No additional packages were installed for Problems 2-4. A fresh clone downloads Flower Photos and official ImageNet VGG16 weights when needed; an existing HW4 flower copy can be reused.

## Methods

Problem 2 adapts the three-block `BasicCNN` in `1_Augmentation_CNN_Classify_flowers.ipynb`. It uses 64-pixel images, batch 64, Adam at 0.001 and 30 epochs, following the course code. `BatchNorm2d` follows every convolution and precedes ReLU. Shared convolution and linear starting values are copied and checked across all four runs. Augmentation changes only training images: left-right flips, small rotations and mild brightness/contrast variation. All complete states, including BatchNorm buffers, are selected by validation loss and reloaded before evaluation.

Problem 3 adapts `3_FineTune_VGG16_PyTorch_Cat_Dog.ipynb` to five flower classes. Images are 150 x 150 with ImageNet normalisation. Stage 1 trains a new five-class head for five epochs with frozen convolution layers. Stage 2 unfreezes only the final convolutional block and trains for three epochs. The course RMSprop settings are retained. Both stages include the incoming model as an epoch-zero selection candidate and use validation loss with patience 3. Exact tensor comparisons verify the frozen layers and saved reloads.

Problem 4 follows Lab 02's class-specific Grad-CAM calculation at VGG16 `features[28]`, a single `Conv2d`. It uses the predicted class of the first sorted validation daisy, selected before prediction. The raw map is 9 x 9 and is enlarged to 150 x 150 for display. Direct gradients avoid backward-hook issues with in-place ReLU. The forward scores and saved model state remain unchanged.

## Saved files and checks

Scripts `01_tiger_transformations.py` through `04_vgg16_gradcam.py` reproduce each problem. Shared modules are `hw5_common.py`, `hw5_basic_cnn.py` and `hw5_vgg.py`. `run_all.py` executes the combined notebook and exports its HTML.

Code, the tiger asset, split records, histories, predictions, numerical maps, figures, notebooks and HTML reports are tracked in Git. Datasets, downloaded ImageNet weights and large trained `.pth` checkpoints remain local and are ignored. Running the notebooks recreates the checkpoints. Problem 3 writes `outputs/problem3/stage1_vgg16.pth` and `stage2_vgg16.pth`; Problem 4 reads the latter.

`python check_submission.py` verifies saved artifacts without retraining: successful combined-notebook execution and its HTML file, split separation and coverage, recomputed test metrics, selected checkpoints, shared CNN starting weights, BatchNorm buffers, frozen VGG16 layers, gallery predictions and Grad-CAM records. It writes `outputs/submission_validation.json` only when every check passes.

`outputs/execution_log.json` records the notebook runs. Full measured results and conclusions are in the notebooks and `outputs/problem1/` through `outputs/problem4/`. One seed and one split were used; variation across seeds was not measured.
