# AAISPT: Adaptive AI-assisted Single Particle Trajectory Analysis Framework
# Overview
AAISPT is a flexible and robust deep learning-based framework for automated, accurate, and scalable analysis of multi-dimensional heterogeneous single particle tracking (SPT) trajectories. It integrates classical 3D SPT methods with modern adaptive deep learning models to enable dynamic segmentation and motion-state classification across a variety of biological imaging modalities and spatial dimensions (2D–5D). The AAISPT is designed to address the challenges in analyzing complex, multi-dimensional SPT data acquired from live-cell imaging. Built upon a series of neural architectures including BiLSTM and Transformer encoders, the framework enables precise motion state inference and dynamic segmentation of heterogeneous trajectories, with strong adaptability across different imaging setups and spatial dimensions.


# Key Features
## **Multi-dimensional Adaptive Trajectory Segmentation**  
The adaptive Bidirectional Long Short-Term Memory (BiLSTM) network is used to segment heterogeneous trajectories into subsegments with distinct diffusion behaviors. The model seamlessly adapts to various input dimensions and imaging conditions, handles variable-length trajectories with multiple change points, and achieves high accracy in change point detection.

## **Pretrained Classification Model with Transformer Encoder**  
A pretrained model integrating an adaptive projection layer with a Transformer encoder is developed to capture the mapping between trajectory features and motion states through a multi-task learning strategy. The model exhibits strong generalization capability across various trajectory types and achieves high performance in motion classification tasks.

## **Trajectory Classification Application with Fine-Tuning**  
By incorporating a fine-tuning strategy, the pretrained model can be efficiently transferred to diverse downstream biological systems, significantly reducing the need for large amounts of labeled data while maintaining high accuracy and adaptability.

# Installation  
## Clone this repo
git clone https://github.com/DongliangSong/AAISPT.git  
cd AAISPT  

## Install dependencies:  
pip install -r requirements.txt   

## Make sure the following core libraries are installed:  
h5py==3.11.0  
hmmlearn==0.3.2  
imbalanced_learn==0.12.3  
imblearn==0.0  
joblib==1.3.2  
matplotlib==3.8.3  
numba==0.59.0   
numpy==1.22.4   
pandas==1.4.4  
PyYAML==6.0  
scikit_learn==1.4.2  
scikit_optimize==0.10.2  
scipy==1.10.0  
seaborn==0.11.2  
shap==0.42.1  
torch==1.12.1  
tqdm==4.66.1  
umap==0.1.1  
umap_learn==0.5.7  

# Simplified Usage  
## Trajectory Simulation  
First, use `.\AAISPT\ij-trajectory-classifier\src\main\java\de\biomedical_imaging\ij\trajectory_classifier\help\GenerateTrainingSet.java` to generate 2D/3D trajectories. Set the parameters in the program and run it directly. Parameters include file storage path, number of diffusion modes, dimensionality, time resolution, diffusion coefficient, signal-to-noise ratio, etc.  

## Generating Trajectory Segmentation Dataset (under .\AAISPT\Gen_trace) 
Use `\Gen_trace_seg.py` to generate the trajectory segmentation dataset. The input is the .txt file from the previous step, and the output is {mode}.mat. Note that for segmentation tasks, trajectories of uniform length are sufficient. If needed, you can further use `\Rotation_addnoise.py` to add noise to rotation angles to match angular resolution errors under different signal-to-noise ratios, saving the results as addnoise_{mode}.mat.  

## Generating Trajectory Classification Dataset (under .\AAISPT\Gen_trace) 
Based on the simulated trajectories in .txt format from the previous step, use `\Gen_trace_cls_varlen.py` to generate trajectories with varying lengths and multiple diffusion modes. Subsequently, apply `\Rotation_addnoise.py` to add noise to rotation step angles to match angular resolution errors under different signal-to-noise ratios, saving the results as addnoise_{mode}.mat. Finally, process the data with `\Gen_varLD_trace_cls.py` to generate trajectories with varying dimensions, saving the results as varLD_{mode}.mat.  

## Diffusion and Rotation Feature Extraction (under .\AAISPT\Feature_extraction) 
Use `\Rolling_extract_features.py` to extract features from the varLD_{mode}.mat files using a sliding window approach. You can use predefined fixed window lengths and step sizes or variable lengths and step sizes, with results saved as Roll_{mode}-feature.mat. To handle outliers in trajectory features, use `\Feature_denoise.py` to remove them based on interquartile range, saving the results as denoise_{mode}_feature.mat.   

## Experimental Data Augmentation (under .\AAISPT\Data_enhance) 
For experimental datasets obtained from different teams, perform data augmentation using methods such as translation, interpolation, adding Gaussian noise, slicing, reversal, or their combinations. Refer to ` \Endocytosis\endocytosis_augmentation.py`, `\Janus_particle\Janus_augmentation.py`, and ` \Phase_separation\phase_separation_augmentation.py`. You can specify parameters for each method and the number of augmentations. To downsample trajectories exceeding a specific length to reduce redundancy, use `\Data_downsample.py`. For augmented or downsampled trajectories, extract diffusion and rotation features using a sliding window approach with `.\AAISPT\Feature_extraction\Exp_feature.py`, saving the results as exp_feature.mat.  

## Trajectory Segmentation Model Training and Validation (under .\AAISPT\Traj_seg\Seg_Adap_BiLSTM_BCE)  
First, preprocess the trajectory segmentation dataset using `\Data_Preprocessing.py` to obtain norm_{mode}.mat. Then, set training hyperparameters (e.g., number of iterations, batch size, learning rate) and model hyperparameters in `\main.py` to train the adaptive trajectory segmentation model. Evaluate the trained model using `\test.py` and `\Metric_evaluation.py` to calculate errors on the test set. Use `\Seg.py` to identify switching points in unknown data and obtain statistical parameters such as segment velocities, diffusion coefficients, and anomaly indices.  

## Trajectory Classification (under .\AAISPT\Traj_cls\AdapTransformer)  
Preprocess (standardize) and train/validate the sliding feature dataset using `\main.py`. Unlike the trajectory segmentation model, this uses a multi-task learning approach for training and validation, enabling the model to handle various input dimensions and output modes. For testing, use `\test.py`. For downstream classification tasks, preprocess new data with `\Fine_tuning\Preprocessing.py` and fine-tune the model using `\Fine_tuning\Fine_tuning.py`.  

# Contact   
For questions, please contact: songnwu@163.com (Xiamen University). 

# License   
This project is licensed under the MIT License. See the LICENSE file for details.

