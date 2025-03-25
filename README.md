# Vehicle Tracking

This project implements a vehicle tracking algorithm to follow a car through a sequence of images. The goal is to build a robust, scalable solution that leverages deep learning and custom tracking logic to operate effectively even in low-data regimes.

---

## Table of Contents

- [Data](#data)
- [Implementation Guidelines](#implementation-guidelines)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Appendix: Disallowed OpenCV Methods](#appendix-disallowed-opencv-methods)

---

## Data

- **Images:**  
  A sequence of images is provided in the `/data` directory, drawn from the VOT 2014 challenge dataset.
  
- **Annotations:**  
  - `groundtruth.txt`: Contains bounding boxes for each frame. Each bounding box is given as a comma-separated list of 8 floating-point numbers in (x, y) order representing the four corners (first, second, third, and fourth).

- **Augmentations:**  
  - The detector is trained by extracting feature data from the validation set and then running augmentation algorithms to generate synthetic data to further train the model in a low-data regime manner.

- **In-Distribution Test Set:**  
  - The tracker is successful in detecting vehicles in an in-distribution test set, provided in the data/ directories as movie files.

---

## Implementation Details

- **Algorithmic Approach**

The vehicle tracker is separated into two components, a detector model and a tracking algorithm. 

The detector is a deeply learned model using a pretrained SegFormer architecture. This architecture generates per-pixel segmentation masks of vehicles. I choose this model as it provides more accurate detection at the pixel level, whereas using a model like YOLO-X will only provide bounding boxes. With per-pixel segmentation, detections with partial occlusions as seen in the sample data (eg. heavy tree occlusion) can be precisely trained. It should be acknowledged that segmentation models are not as lightweights and performant as a bounding box proposal model, however the focus of this project is batch processing of sample data, not necessarily real-time applications.

- Input Features
  - Current RGB pixel data [ 3 channel, bottom half of input field ]
  - Gray Scale delta from previous image in the input sequence [ 1 channel, top half of input field ]
  - Gray Scale prior frame data [ 1 channel, top half of input field ]
  - Gray Scale current frame data [ 1 channel, top half of input field ]
  
The detector model uses various features as described above, all merged into one fusion SegFormer model. The additional pixel data beyond the current frame assist with sequence modeling; the model uses prior frame data to better segment the current image by understanding what is currently moving in the scene, focusing less on color information, and more on structure changes. 

The features extracted above worked well. Originally I tried single RGB pixel data, which did not perform significantly well. I then tried using multiple frames of RGB data, which did perform better; this apparently helped the model to understand the changes in the scene to improve detections. I also trained the model with Optical Flow features. This provided a well performing model, improving IoU scoring to 0.94 during validation. However this approach is slower to train as the OpenCV library implementation runs on CPU. As the alternative, the final model used image differencing with gray scale features which provided similar performance as optical flow features, but with zero additional training overhead.

The tracker algorithm performs the object tracking and utilizes the detector model to highlight pixel regions in an image. It works in these phases below.

- Tracking Phases
   - Detection Phase across entire sequence
   - Refine detected regions
   - Object trajectory generation using keyframes and correspondence matching
   - Output visualization

In the detection phase, as an object moves across the field of view, the detector will generate regions and the tracker will associate these detections with objects. The regions are clustered together and assigned to an object using spatial overlap from previous frame data, but could also be clustered by visual similarity and movement vectors to disambiguate them from overlapping objects.

Once the entire sequence has been processed, a refinement phase is performed. The detection model generates regions which tend to over-predict the boundary of the actual object under certain conditions. The refinement phase runs through all frames of each objects tracking history and performs a sliding window refinement process. Cross-correlation scores are used to compare the object boundaries between frames and reduce the size of the boundaries to be temporally consistent. This eliminates spurious over-predictions leading to far improved visual tracking stability.

The object trajectories are determined after the detection and refinement phases. The goal of this phases to generate a smooth trajectory based on the object detections made in the first phase. The idea for this is predicated upon this important concept: vehicles tend to have fixed shapes and thus consistent object sizing across a sequence of images. Given this information, the regions are clustered into keyframe groups, which are sequences where the vehicle has a stable visual size. If the visual size becomes unstable, this is likely due to an occlusion where detections become sparse or non-existent. To overcome the occlusion, regions with unstable tracking are compared to nearby keyframes and correspondence matching is performed using cross correlation to determine the mapping of detected regions to other regions in the sequence. By determining how the detections across frame map to each other, partial detections can thus rebuild the object's true size with the keyframe object size information. Object size stability and position can then be reconstructed if there are partial or missing detections. After all frames have been assigned correctly positioned and sized object regions, a final smoothing pass is performed on the region boxes to finalizing the objects tracking trajectory.

Once trajectories are completed, output visualization is performed. If there are labels available, a matching algorithm is run to associate labels with objects by spatial proximity. IoU scoring can then be calculated to determine how well the trajactories perform. Smoothing tends to increase the IoU score by a few relative percentage points. Multiple vehicles can be tracked as the tracking algorithm can assign spatially distant regions to different objects, however as mentioned this can be refined by using visual similarity and other cues.

- **Training Approach**

PyTorch Lightning was used as the deep learning framework. The HuggingFace package was used to import models and pre-trained weights. The SegFormer model I choose was pretrained on CityScapes, a dataset that has vehicle segmentation as a primary task. The model consists of approxiamately 4M parameters, 400k which are in the final deocder module head. This module was fine-tuned while the rest of the model weights were frozen. This essentially keeps the model's early layers focused on vehicle detection without overfitting to the new training set that has new visual features. Surprisingly the model adaptation worked well with image differencing and gray scale features which are out-of-distribution of the original training set. The model can be trained in about 5 minutes on a single T4 GPU, within 20-40 epochs depending on the seeded model.

This project's goal was to avoid using external data and only discover ways to train a model in a principled manner with only one data sequence. Towards this goal, data augmentation was a significant focus. To start, the sample data provided was segmented using a Segment Anything model to generate vehicle labels. The labels were reviewed and bad labels were marked to exclude them from training. These new labels are extracted vehicle pixels from each image, with no background pixels, which improves per-pixel vehicle detection accuracy. The label and background layers are then used for image augmentation during training. The training set is a completely synthetic mixture of random vehicle labels and backgrounds; each image contains one vehicle moving in random directions and positions across the image as well as having various size, color, contrast, hue, rotation, and shearing applied to but background and vehicle pixels. This acts as a form of regularization to help generalize the model and avoid overfitting to the sample data. 

The model is trained using various objective functions. A hyperparameter search was done using various combinations of loss functions with different tuning values. The final combination that worked best was BCE combined with a focal and dice loss, in a addition to a total variation loss that acted as a regularizing term. This helped enforce detected regions to be dense, essential for high quality and condifident predictions.

- **What didn't work**

For training, some loss functions didn't work well, like the IoU loss. This loss seemed to be unstable and lead to poor results. Also I tried a consistency loss that forced similarity between predictions in the segmenetation model's sequence based outputs. This loss seemed to act as too much of a regularizing factor that the model's training could not proceed past a certain point; I believe the model couldn't determine whether to grow or shrink the detected regions leading to it remaining stuck in the current local optimization minima. 

Single frame detection segmentation alone did not offer good results. Using multiple frames as inputs did help, however it lead to heavy overpredictions of the vehicle boundaries as more frames were added (up to 5 were used). Optical flow features absolutely helped train the model to understand motion, however this seemed too powerful so image differencing was used instead and worked well.

Occlusion augmentation was implemented; various vehicle pixels were clipped with stripping patterns and circular patterns during augmentation. The results of this test were inconclusive - there were many model adjustments happening that could have conflated the experimental results. I imagine this would help with occlusion generalization but would increase the training time to allow the model to adapt to this further regularization. 

In terms of the tracker, trying to simply smoothly interpolate the object detections did not work well. Sliding window averages were poor substitutes for a more targeted approach focused on repairing the noisy areas of the sequence, and then doing a final phase of filtering. Refining the detection pixels could lead to decent outcomes, but looking at the samples with and without the detection refinement made it clear it was a major win, in particular with very small detections of distant objects that tend to be noisier. 

- **Limitations**

The model has limitations based on its limited training set. I would imagine it will fail in out-of-distribution data, such as scenes were cars are detected from the front or from directly behind, or against backgrounds such as urban settings, or different vehicles form factors like trucks. The model does detect cars on similar test scenes based on experiments. Also multiple vehicle generation shows some gaps in detections particularly in regions where the target vehicle has low contrast with the background. The model may be overfitting to the 'roadway' as a specific detection signal.

Multiple vehicle tracking does work but is not robust. If vehicles overlap, their regions are clustered together and not disambiguated. This leads to inappropriately assigned object labels as the vehicles move close to each other. As mentioned, visual similarity, speed, and direction of motion can be used to disambiguate the regions to better track overlapping objects.

---

## Project Structure

A suggested project layout:

```
vehicle-tracking/
├── README.md
├── requirements.txt           # or environment.yml / poetry.lock
├── src/
│   ├── __init__.py
│   ├── augment_images.py      # helper script used to extract new training/validation data from the sample dataset
│   ├── augment.py             # data augmentation toolkit
│   ├── detector.py            # HuggingFace/PyTorch Lightning framework for model definition and training
│   ├── evaluate.py            # evaluation script for the detector model only
│   ├── loader.py              # PyTorch Dataset/Dataloader definitions
│   ├── losses.py              # Various ML loss/objective functions
│   ├── tracker.py             # main tracking algorithm and evaluation using the detection model
│   └── tools.py               # helper functions, e.g., for data I/O, visualization, mathematical utilities
│   
├── data/
│   ├── groundtruth.txt        # bounding box annotations
│   ├── *.label                # per-frame attribute files
│   ├── *.jpg                  # sample data files used as a validation/evaluation dataset
│   └── *.mp4/*.avi            # sample data files used as in-distribution test set 
│
├── test/
│   ├── __init__.py
│   └── *.py.                  # unit tests and integration tests
│
└── dashboard.py               # StreamLit Dashboard
```

---

## Usage

**Installation:**  
Provide instructions on how to set up your environment. For example:

```bash
# Using pip
pip install -r requirements.txt

# or using conda
conda env create -f environment.yml
conda activate vehicle-tracking
```

**Training the Detector:**

There are prebuilt models already available in the `./checkpoints` folder. The detector model can be custom trained using the following command. There currently are no command arguments for this.

```bash
python -m src.detector
```

**Running the Tracker:**  

To run the tracker with the sample data in the `./data/` directory.

```bash
python -m src.tracker
```

The tracker supports various command line options. You can run the following command to see the available options.

```bash
python -m src.tracker -h
```

To run the tracker with all visualization options enabled:

```bash
python -m src.tracker --enable_segmentation --enable_smooth_tracking --enable_detection_tracking
```

For some extra fun, you can add extra synthetic vehicles for additional tracking experiments. This only works with the default dataset, not with custom loaded data.

```bash
python -m src.tracker --num_vehicles 4
```

You can specify a directory for input images. If a `groundtruth.txt` is available in this directory, it will be used to label the 1st object and produce an IoU score.

```bash
python -m src.tracker --dir DATA_DIRECTORY
```

*Note*: When using the default data set, detections are cached to a file. This speeds up tracking generation if there options are changed. To rerun the detections, run the command below. 
When using custom datasets or other augmentation parameters, the cache file is ignored.

```bash
python -m src.tracker --dir DATA_DIRECTORY
```

**Dashboard Utility**

A StreamLit Dashboard is provided. To run:

```bash
streamlit run dashboard.py
```

**Testing:**  

Tests are *almost* non-existent. What is provided here is a sample of what more tests could look like.

Run tests with:

```bash
PYTHONPATH=. pytest
```

---

## Approach and Discussion

- **Algorithmic Approach:**
  - Briefly describe the algorithm you implemented.
  - Explain how your deep learning model is used (e.g., feature extraction, regression, etc.).
  - Discuss any innovative ideas or transformations (e.g., handling occlusion, leveraging temporal context).

- **Other Considered Approaches:**
  - List alternative methods or architectures you considered.
  - Mention any experiments that did not work as expected and what you learned.

- **Potential Improvements:**
  - Outline future enhancements that would improve the model’s robustness or efficiency.
  - Discuss scalability considerations and how the approach could generalize to larger datasets.

- **Pretrained Models:**
  - If applicable, explain which pretrained models you used and why they were chosen.

---

## Appendix: Disallowed OpenCV Methods

This project was created with the intention to avoid using the following (and similar) methods as the core of the tracking solution:

- `cv::Tracker` and its subclasses
- `cv::KalmanFilter`
- `cv::BackgroundSubtractor` and its subclasses
- `cv::findContours`
- `cv::matchTemplate`
- `cv::calcOpticalFlowPyrLK`
- `cv::calOpticalFlowFarneback`
- `cv::goodFeaturesToTrack`

Note: The code utilized `cv::calOpticalFlowFarneback` to generate features used to train a detector model as a auxiliary set. However these features are not used during inference. Furthermore, image differencing was just as useful thus optical flow was not necessary for the
success of the detector.

---
