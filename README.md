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

<table class="MsoTableGrid" border="1" cellspacing="0" style="border-collapse:collapse;border:none;mso-border-left-alt:0.5000pt solid windowtext;
mso-border-top-alt:0.5000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;mso-border-bottom-alt:0.5000pt solid windowtext;
mso-border-insideh:0.5000pt solid windowtext;mso-border-insidev:0.5000pt solid windowtext;mso-padding-alt:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;"><tbody><tr><td width="93" valign="top" style="width:55.8000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:1.0000pt solid windowtext;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;"><font face="Calibri">Categories</font></span><span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="50" valign="top" style="width:30.3500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:1.0000pt solid windowtext;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;"><font face="Calibri">Ckpt</font></span><span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="154" valign="top" colspan="3" style="width:92.8000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:1.0000pt solid windowtext;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;"><font face="Calibri">Binary Cls</font></span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="137" valign="top" colspan="3" style="width:82.3500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:1.0000pt solid windowtext;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;"><font face="Calibri">Multi-label Cls</font></span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="137" valign="top" colspan="3" style="width:82.3500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:1.0000pt solid windowtext;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;"><font face="Calibri">Image Grounding</font></span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="137" valign="top" colspan="3" style="width:82.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:1.0000pt solid windowtext;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;"><font face="Calibri">Text Grounding</font></span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td></tr><tr style="height:16.4000pt;"><td width="93" valign="top" style="width:55.8000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;"><font face="Calibri">Methods</font></span><span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="50" valign="top" style="width:30.3500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:1.0000pt;"><o:p>&nbsp;</o:p></span></p></td><td width="63" valign="top" style="width:37.9500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">AUC</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">EER</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">ACC</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">mAP</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:1.0000pt solid windowtext;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">CF1</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:1.0000pt solid windowtext;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">OF1</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:1.0000pt solid windowtext;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">IoUmean</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:1.0000pt solid windowtext;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">IoU50</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:1.0000pt solid windowtext;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">IoU75</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:1.0000pt solid windowtext;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">Precision</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.5000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:1.0000pt solid windowtext;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">Recall</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.5000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:1.0000pt solid windowtext;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">F1</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td></tr><tr><td width="93" valign="top" style="width:55.8000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;"><font face="Calibri">CLIP</font></span><span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="50" valign="top" style="width:30.3500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:1.0000pt;"><o:p>&nbsp;</o:p></span></p></td><td width="63" valign="top" style="width:37.9500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">78.33</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">29.30</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">72.03</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">62.47</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">54.91</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">55.06</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">44.58</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">44.52</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">44.51</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">56.68</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.5000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">31.17</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.5000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">40.22</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td></tr><tr><td width="93" valign="top" style="width:55.8000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;"><font face="Calibri">ViLT</font></span><span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="50" valign="top" style="width:30.3500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:1.0000pt;"><o:p>&nbsp;</o:p></span></p></td><td width="63" valign="top" style="width:37.9500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">85.73</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">22.41</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">79.04</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">72.66</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">66.17</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">66.11</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">54.87</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">59.14</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">41.48</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">66.61</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.5000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">46.99</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.5000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">55.11</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td></tr><tr><td width="93" valign="top" style="width:55.8000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;"><font face="Calibri">HAMMER</font></span><span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="50" valign="top" style="width:30.3500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:1.0000pt;"><o:p>&nbsp;</o:p></span></p></td><td width="63" valign="top" style="width:37.9500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">90.62</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">17.11</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">84.18</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">83.20</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">76.71</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">77.01</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">74.64</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">81.69</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">74.56</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><b><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-weight:bold;font-size:10.5000pt;
mso-font-kerning:0.0000pt;">74.74</span></b><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.5000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">63.46</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.5000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">68.64</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td></tr><tr><td width="93" valign="top" style="width:55.8000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;"><font face="Calibri">Ours</font></span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="50" valign="top" style="width:30.3500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:1.0000pt;"><o:p>&nbsp;</o:p></span></p></td><td width="63" valign="top" style="width:37.9500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><b><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-weight:bold;font-size:10.5000pt;
mso-font-kerning:0.0000pt;">93.59</span></b><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><b><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-weight:bold;font-size:10.5000pt;
mso-font-kerning:0.0000pt;">14.22</span></b><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><b><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-weight:bold;font-size:10.5000pt;
mso-font-kerning:0.0000pt;">86.92</span></b><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><b><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-weight:bold;font-size:10.5000pt;
mso-font-kerning:0.0000pt;">85.12</span></b><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><b><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-weight:bold;font-size:10.5000pt;
mso-font-kerning:0.0000pt;">78.67</span></b><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><b><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-weight:bold;font-size:10.5000pt;
mso-font-kerning:0.0000pt;">79.28</span></b><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><b><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-weight:bold;font-size:10.5000pt;
mso-font-kerning:0.0000pt;">76.47</span></b><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><b><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-weight:bold;font-size:10.5000pt;
mso-font-kerning:0.0000pt;">83.38</span></b><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><b><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-weight:bold;font-size:10.5000pt;
mso-font-kerning:0.0000pt;">76.45</span></b><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.4500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-size:10.5000pt;mso-font-kerning:0.0000pt;">72.54</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.5000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><b><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-weight:bold;font-size:10.5000pt;
mso-font-kerning:0.0000pt;">70.63</span></b><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.5000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="p" align="center" style="margin-right:0.0000pt;margin-left:0.0000pt;mso-para-margin-right:0.0000gd;
mso-para-margin-left:0.0000gd;mso-pagination:widow-orphan;text-align:center;"><b><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
color:rgb(0,0,0);font-weight:bold;font-size:10.5000pt;
mso-font-kerning:0.0000pt;">71.57</span></b><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:0.0000pt;"><o:p></o:p></span></p></td></tr></tbody></table>

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




