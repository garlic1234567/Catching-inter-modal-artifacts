# Catching Inter-Modal Artifacts: A Cross-Modal Framework for Temporal Forgery Localization
This is the repository for the paper "Catching Inter-Modal Artifacts: A Cross-Modal Framework for Temporal Forgery Localization".

## Dependency
Since we build on the UMMAFormer (Zhang et al.,2023) framework for our TFL framework, please refer to [UMMAFormer repository](https://github.com/ymhzyj/UMMAFormer) for detailed dependencies required to reproduce our Cross-Modal Framework designed for the TFL task in the [TFL](./TFL) folder. <br> <br>
For experiments validating the generalizability of the proposed modules, since we replace the original alignment and fusion operation in HAMMER with our BRM and DFFM, respectively, please refer to [HAMMER repository](https://github.com/rshaojimmy/MultiModal-DeepFake) for detailed installation instructions.

## Dataset
We conduct experiments on the following datasets:
1. TFL Benchmark Dataset Lav-DF: This dataset is used to evaluate our model for Temporal Forgery Localization (TFL). We preprocess this dataset in the same manner as our baseline model [UMMAFormer](https://github.com/ymhzyj/UMMAFormer).
2. DGM4 Dataset: Used to validate the cross-modal generalizability of the proposed BRM and DFFM modules. All the configurations are the same to [HAMMER](https://github.com/rshaojimmy/MultiModal-DeepFake).

## Reproduction on DGM4
In the [reproduction](./reproduction) folder, we provide the implementation for replicating the experimental results of CLIP, ViLT, and HAMMER on the DGM4 dataset, as reported in our paper.

<table width="722.00" border="0" cellpadding="0" cellspacing="0" style="width:433.20pt;border-collapse:collapse;table-layout:fixed;">
   <colgroup><col width="93.00" style="mso-width-source:userset;mso-width-alt:2720;">
   <col width="50.58" style="mso-width-source:userset;mso-width-alt:1479;">
   <col width="63.25" style="mso-width-source:userset;mso-width-alt:1850;">
   <col width="51.50" style="mso-width-source:userset;mso-width-alt:1506;">
   <col width="51.67" style="mso-width-source:userset;mso-width-alt:1511;">
   <col width="45.75" span="7" style="mso-width-source:userset;mso-width-alt:1338;">
   <col width="45.83" style="mso-width-source:userset;mso-width-alt:1340;">
   <col width="45.92" style="mso-width-source:userset;mso-width-alt:1343;">
   </colgroup><tbody><tr height="25.25" style="height:15.15pt;mso-height-source:userset;mso-height-alt:303;">
    <td class="xl65" height="25.25" width="93.00" style="height:15.15pt;width:55.80pt;" x:str="">Categories</td>
    <td class="xl65" width="50.58" style="width:30.35pt;" x:str="">Ckpt</td>
    <td class="xl65" width="166.42" colspan="3" style="width:99.85pt;border-right:1.0pt solid windowtext;border-bottom:1.0pt solid windowtext;" x:str="">Binary Cls</td>
    <td class="xl65" width="137.25" colspan="3" style="width:82.35pt;border-right:1.0pt solid windowtext;border-bottom:1.0pt solid windowtext;" x:str="">Multi-label Cls</td>
    <td class="xl65" width="137.25" colspan="3" style="width:82.35pt;border-right:1.0pt solid windowtext;border-bottom:1.0pt solid windowtext;" x:str="">Image Grounding</td>
    <td class="xl65" width="137.50" colspan="3" style="width:82.50pt;border-right:1.0pt solid windowtext;border-bottom:1.0pt solid windowtext;" x:str="">Text Grounding</td>
   </tr>
   <tr height="27.33" style="height:16.40pt;mso-height-source:userset;mso-height-alt:328;">
    <td class="xl66" height="27.33" style="height:16.40pt;" x:str="">Methods</td>
    <td class="xl66"></td>
    <td class="xl67" x:str="">AUC</td>
    <td class="xl67" x:str="">EER</td>
    <td class="xl67" x:str="">ACC</td>
    <td class="xl67" x:str="">mAP</td>
    <td class="xl68" x:str="">CF1</td>
    <td class="xl68" x:str="">OF1</td>
    <td class="xl68" x:str="">IoUmean</td>
    <td class="xl68" x:str="">IoU50</td>
    <td class="xl68" x:str="">IoU75</td>
    <td class="xl68" x:str="">Precision</td>
    <td class="xl68" x:str="">Recall</td>
    <td class="xl68" x:str="">F1</td>
   </tr>
   <tr height="481.25" style="height:288.75pt;">
    <td class="xl66" height="481.25" style="height:288.75pt;" x:str="">CLIP</td>
    <td class="xl66" x:str="">[ckpt](https://1drv.ms/u/c/39d9599dfa883d20/EaaEobf2eGdOvBky9Px3V1MBuwa7oeVlbg3XCQDrzb62cA)</td>
    <td class="xl67" x:num="">78.33</td>
    <td class="xl67" x:num="">29.3</td>
    <td class="xl67" x:num="">72.03</td>
    <td class="xl67" x:num="62.469999999999999">62.5</td>
    <td class="xl67" x:num="54.909999999999997">54.9</td>
    <td class="xl67" x:num="55.060000000000002">55.1</td>
    <td class="xl67" x:num="44.579999999999998">44.6</td>
    <td class="xl67" x:num="44.520000000000003">44.5</td>
    <td class="xl67" x:num="44.509999999999998">44.5</td>
    <td class="xl67" x:num="56.68">56.7</td>
    <td class="xl67" x:num="31.170000000000002">31.2</td>
    <td class="xl67" x:num="40.219999999999999">40.2</td>
   </tr>
   <tr height="25.25" style="height:15.15pt;">
    <td class="xl66" height="25.25" style="height:15.15pt;" x:str="">ViLT</td>
    <td class="xl66"></td>
    <td class="xl67" x:num="">85.73</td>
    <td class="xl67" x:num="">22.41</td>
    <td class="xl67" x:num="">79.04</td>
    <td class="xl67" x:num="72.659999999999997">72.7</td>
    <td class="xl67" x:num="66.170000000000002">66.2</td>
    <td class="xl67" x:num="66.109999999999999">66.1</td>
    <td class="xl67" x:num="54.869999999999997">54.9</td>
    <td class="xl67" x:num="59.140000000000001">59.1</td>
    <td class="xl67" x:num="41.479999999999997">41.5</td>
    <td class="xl67" x:num="66.609999999999999">66.6</td>
    <td class="xl67" x:num="46.990000000000002">47</td>
    <td class="xl67" x:num="55.109999999999999">55.1</td>
   </tr>
   <tr height="25.25" style="height:15.15pt;">
    <td class="xl66" height="25.25" style="height:15.15pt;" x:str="">HAMMER</td>
    <td class="xl66"></td>
    <td class="xl67" x:num="">90.62</td>
    <td class="xl67" x:num="">17.11</td>
    <td class="xl67" x:num="">84.18</td>
    <td class="xl67" x:num="">83.2</td>
    <td class="xl67" x:num="76.709999999999994">76.7</td>
    <td class="xl67" x:num="77.010000000000005">77</td>
    <td class="xl67" x:num="74.640000000000001">74.6</td>
    <td class="xl67" x:num="81.689999999999998">81.7</td>
    <td class="xl67" x:num="74.560000000000002">74.6</td>
    <td class="xl69" x:num="74.739999999999995">74.7</td>
    <td class="xl67" x:num="63.460000000000001">63.5</td>
    <td class="xl67" x:num="68.640000000000001">68.6</td>
   </tr>
   <tr height="25.25" style="height:15.15pt;">
    <td class="xl66" height="25.25" style="height:15.15pt;" x:str="">Ours</td>
    <td class="xl66"></td>
    <td class="xl69" x:num="">93.59</td>
    <td class="xl69" x:num="">14.22</td>
    <td class="xl69" x:num="">86.92</td>
    <td class="xl69" x:num="85.120000000000005">85.1</td>
    <td class="xl69" x:num="78.670000000000002">78.7</td>
    <td class="xl69" x:num="79.280000000000001">79.3</td>
    <td class="xl69" x:num="76.469999999999999">76.5</td>
    <td class="xl69" x:num="83.379999999999995">83.4</td>
    <td class="xl69" x:num="76.450000000000003">76.5</td>
    <td class="xl67" x:num="72.540000000000006">72.5</td>
    <td class="xl69" x:num="70.629999999999995">70.6</td>
    <td class="xl69" x:num="71.569999999999993">71.6</td>
   </tr>
   <tr height="24" style="height:14.40pt;mso-height-source:userset;mso-height-alt:288;">
    <td class="xl70" height="24" colspan="14" style="height:14.40pt;border-right:none;border-bottom:none;" x:str=""><span style="mso-spacerun:yes;">&nbsp;</span></td>
   </tr>
   <!--[if supportMisalignedColumns]-->
    <tr width="0" style="display:none;">
     <td width="93" style="width:56;"></td>
     <td width="51" style="width:30;"></td>
     <td width="63" style="width:38;"></td>
     <td width="52" style="width:31;"></td>
     <td width="52" style="width:31;"></td>
     <td width="46" style="width:27;"></td>
     <td width="46" style="width:28;"></td>
     <td width="46" style="width:28;"></td>
    </tr>
   <!--[endif]-->
  </tbody></table>
  
### Experimental Setup
- Hardware: All the three methods are reproduced on 2 A6000 GPUs.
- HAMMER: For reproducing HAMMER, we directly use the official implementation of HAMMER provided in [HAMMER repository](https://github.com/rshaojimmy/MultiModal-DeepFake).
- CLIP and ViLT: The reproduction of CLIP and ViLT is based on the methodology described in Section 5.1 of the paper *Detecting and Grounding Multi-Modal Media Manipulation*. 

## Training and Testing for TFL
### Training
To train our model for Temporal Forgery Localization (TFL):
```
cd /TFL
python ./train.py -config ./configs/lavdf_tsn_byola.yaml
```
### Testing
To test our model for TFL:
```
cd /TFL
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
cd /MultiModal-DeepFake-main
sh my_test.sh
```




