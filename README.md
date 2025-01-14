# Catching Inter-Modal Artifacts: A Cross-Modal Framework for Temporal Forgery Localization
This is the repository for the paper "Catching Inter-Modal Artifacts: A Cross-Modal Framework for Temporal Forgery Localization".

# Dependency
The following dependencies are required to run the code:
- Linux
- Python 3.8
- Pytorch 1.13
- Numpy
- PyYaml
- Pandas
- h5py
- joblib
- einops

## Dataset
We conduct experiments on the following datasets:
1. TFL Benchmark Dataset Lav-DF: This dataset is used to evaluate our model for Temporal Forgery Localization (TFL). We preprocess this dataset in the same manner as our baseline model, UMMAFormer(https://github.com/ymhzyj/UMMAFormer).
2. DGM4 Dataset: Used to validate the cross-modal generalizability of the proposed BRM and DFFM modules. All the configurations are the same to HAMMER(https://github.com/rshaojimmy/MultiModal-DeepFake).

## Reproduction on DGM4
In the [reproduction](./reproduction) folder, we provide the implementation for reproducting the performance of CLIP, ViLT and HAMMER on the DGM4 dataset. 

### Experimental Setup
- Hardware: All the three methods are reproduced on 2 A6000 GPUs.
- HAMMER: For reproducing HAMMER, we directly use the official implementation of HAMMER provided in [HAMMER repository](https://github.com/rshaojimmy/MultiModal-DeepFake).
- CLIP and ViLT: The reproduction of CLIP and ViLT is based on the methodology described in Section 5.1 of the paper *Detecting and Grounding Multi-Modal Media Manipulation*. 

## Training and Testing for TFL
### Training
To train our model for Temporal Forgery Localization (TFL):
```
python ./train.py -config ./configs/lavdf_tsn_byola.yaml
```
### Testing
To test our model for TFL:
```
python ./eval.py ./configs/lavdf_tsn_byola.yaml ./paper_results/your_ckpt_folder/model_best.pth.tar
```
Replace your_ckpt_folder with the folder containing your model checkpoint.

## Training and Testing for DGM4
### Training
To train our model for fake news detection and grounding on the DGM4 dataset, navigate to the repository and run:
```
cd /MultiModal-DeepFake-main
sh my_train.sh
```
### Testing
To test our model:
```
sh my_test.sh
```




