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
git clone https://github.com/DongliangSong/AAISPT.git  
cd AAISPT

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

# Contact   
For questions, please contact: songnwu@163.com (Xiamen University). 

# License   
MIT License. See LICENSE for details. 

