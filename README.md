[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models/graphs/commit-activity)
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models)
<img alt="GitHub watchers" src="https://img.shields.io/github/watchers/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models?style=social"> <img alt="GitHub stars" src="https://img.shields.io/github/stars/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models?style=social"> <img alt="GitHub forks" src="https://img.shields.io/github/forks/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models?style=social">

# <p align=center>`Awesome Remote Sensing Foundation Models`</p>

:star2:**A collection of papers, datasets, benchmarks, code, and pre-trained weights for Remote Sensing Foundation Models (RSFMs).**

## 📢 Latest Updates
:fire::fire::fire: Last Updated on 2024.08.19 :fire::fire::fire:

- **2024.8.19**: Update SpectralEarth.
- **2024.8.08**: Update a survey paper.
- **2024.8.06**: Update MA3E.

## Table of Contents
- **Models**
  - [Remote Sensing Vision Foundation Models](#remote-sensing-vision-foundation-models)
  - [Remote Sensing Vision-Language Foundation Models](#remote-sensing-vision-language-foundation-models)
  - [Remote Sensing Generative Foundation Models](#remote-sensing-generative-foundation-models)
  - [Remote Sensing Vision-Location Foundation Models](#remote-sensing-vision-location-foundation-models)
  - [Remote Sensing Vision-Audio Foundation Models](#remote-sensing-vision-audio-foundation-models)
  - [Remote Sensing Task-specific Foundation Models](#remote-sensing-task-specific-foundation-models)
  - [Remote Sensing Agents](#remote-sensing-agents)
- **Datasets & Benchmarks**
  - [Benchmarks for RSFMs](#benchmarks-for-rSFMs)
  - [(Large-scale) Pre-training Datasets](#large-scale-Pre-training-Datasets)
- **Others**
  - [Relevant Projects](#relevant-projects)
  - [Survey Papers](#survey-papers)
  
## Remote Sensing <ins>Vision</ins> Foundation Models

|Abbreviation|Title|Publication|Paper|Code & Weights|
|:---:|---|:---:|:---:|:---:|
|**GeoKR**|**Geographical Knowledge-Driven Representation Learning for Remote Sensing Images**|TGRS2021|[GeoKR](https://ieeexplore.ieee.org/abstract/document/9559903)|[link](https://github.com/flyakon/Geographical-Knowledge-driven-Representaion-Learning)|
|**-**|**Self-Supervised Learning of Remote Sensing Scene Representations Using Contrastive Multiview Coding**|CVPRW2021|[Paper](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/html/Stojnic_Self-Supervised_Learning_of_Remote_Sensing_Scene_Representations_Using_Contrastive_Multiview_CVPRW_2021_paper.html)|[link](https://github.com/vladan-stojnic/CMC-RSSR)|
|**GASSL**|**Geography-Aware Self-Supervised Learning**|ICCV2021|[GASSL](https://openaccess.thecvf.com/content/ICCV2021/html/Ayush_Geography-Aware_Self-Supervised_Learning_ICCV_2021_paper.html)|[link](https://github.com/sustainlab-group/geography-aware-ssl)|
|**SeCo**|**Seasonal Contrast: Unsupervised Pre-Training From Uncurated Remote Sensing Data**|ICCV2021|[SeCo](https://openaccess.thecvf.com/content/ICCV2021/html/Manas_Seasonal_Contrast_Unsupervised_Pre-Training_From_Uncurated_Remote_Sensing_Data_ICCV_2021_paper.html)|[link](https://github.com/ServiceNow/seasonal-contrast)|
|**DINO-MM**|**Self-supervised Vision Transformers for Joint SAR-optical Representation Learning**|IGARSS2022|[DINO-MM](https://arxiv.org/abs/2204.05381)|[link](https://github.com/zhu-xlab/DINO-MM)|
|**SatMAE**|**SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery**|NeurIPS2022|[SatMAE](https://proceedings.neurips.cc/paper_files/paper/2022/hash/01c561df365429f33fcd7a7faa44c985-Abstract-Conference.html)|[link](https://github.com/sustainlab-group/SatMAE)|
|**RS-BYOL**|**Self-Supervised Learning for Invariant Representations From Multi-Spectral and SAR Images**|JSTARS2022|[RS-BYOL](https://ieeexplore.ieee.org/abstract/document/9880533)|null|
|**GeCo**|**Geographical Supervision Correction for Remote Sensing Representation Learning**|TGRS2022|[GeCo](https://ieeexplore.ieee.org/abstract/document/9869651)|null|
|**RingMo**|**RingMo: A remote sensing foundation model with masked image modeling**|TGRS2022|[RingMo](https://ieeexplore.ieee.org/abstract/document/9844015)|[Code](https://github.com/comeony/RingMo)|
|**RVSA**|**Advancing plain vision transformer toward remote sensing foundation model**|TGRS2022|[RVSA](https://ieeexplore.ieee.org/abstract/document/9956816)|[link](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA)|
|**RSP**|**An Empirical Study of Remote Sensing Pretraining**|TGRS2022|[RSP](https://ieeexplore.ieee.org/abstract/document/9782149)|[link](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA)|
|**MATTER**|**Self-Supervised Material and Texture Representation Learning for Remote Sensing Tasks**|CVPR2022|[MATTER](https://openaccess.thecvf.com/content/CVPR2022/html/Akiva_Self-Supervised_Material_and_Texture_Representation_Learning_for_Remote_Sensing_Tasks_CVPR_2022_paper.html)|null|
|**CSPT**|**Consecutive Pre-Training: A Knowledge Transfer Learning Strategy with Relevant Unlabeled Data for Remote Sensing Domain**|RS2022|[CSPT](https://www.mdpi.com/2072-4292/14/22/5675#)|[link](https://github.com/ZhAnGToNG1/transfer_learning_cspt)|
|**-**|**Self-supervised Vision Transformers for Land-cover Segmentation and Classification**|CVPRW2022|[Paper](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/html/Scheibenreif_Self-Supervised_Vision_Transformers_for_Land-Cover_Segmentation_and_Classification_CVPRW_2022_paper.html)|[link](https://github.com/HSG-AIML/SSLTransformerRS)|
|**BFM**|**A billion-scale foundation model for remote sensing images**|Arxiv2023|[BFM](https://arxiv.org/abs/2304.05215)|null|
|**TOV**|**TOV: The original vision model for optical remote sensing image understanding via self-supervised learning**|JSTARS2023|[TOV](https://ieeexplore.ieee.org/abstract/document/10110958)|[link](https://github.com/GeoX-Lab/G-RSIM/tree/main/TOV_v1)|
|**CMID**|**CMID: A Unified Self-Supervised Learning Framework for Remote Sensing Image Understanding**|TGRS2023|[CMID](https://ieeexplore.ieee.org/abstract/document/10105625)|[link](https://github.com/NJU-LHRS/official-CMID)|
|**RingMo-Sense**|**RingMo-Sense: Remote Sensing Foundation Model for Spatiotemporal Prediction via Spatiotemporal Evolution Disentangling**|TGRS2023|[RingMo-Sense](https://ieeexplore.ieee.org/abstract/document/10254320)|null|
|**IaI-SimCLR**|**Multi-Modal Multi-Objective Contrastive Learning for Sentinel-1/2 Imagery**|CVPRW2023|[IaI-SimCLR](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/html/Prexl_Multi-Modal_Multi-Objective_Contrastive_Learning_for_Sentinel-12_Imagery_CVPRW_2023_paper.html)|null|
|**CACo**|**Change-Aware Sampling and Contrastive Learning for Satellite Images**|CVPR2023|[CACo](https://openaccess.thecvf.com/content/CVPR2023/html/Mall_Change-Aware_Sampling_and_Contrastive_Learning_for_Satellite_Images_CVPR_2023_paper.html)|[link](https://github.com/utkarshmall13/CACo)|
|**SatLas**|**SatlasPretrain: A Large-Scale Dataset for Remote Sensing Image Understanding**|ICCV2023|[SatLas](https://arxiv.org/abs/2211.15660)|[link](https://github.com/allenai/satlas)|
|**GFM**|**Towards Geospatial Foundation Models via Continual Pretraining**|ICCV2023|[GFM](https://arxiv.org/abs/2302.04476)|[link](https://github.com/mmendiet/GFM)|
|**Scale-MAE**|**Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning**|ICCV2023|[Scale-MAE](https://arxiv.org/abs/2212.14532)|[link](https://github.com/bair-climate-initiative/scale-mae)|
|**DINO-MC**|**DINO-MC: Self-supervised Contrastive Learning for Remote Sensing Imagery with Multi-sized Local Crops**|Arxiv2023|[DINO-MC](https://arxiv.org/abs/2303.06670)|[link](https://github.com/WennyXY/DINO-MC)|
|**CROMA**|**CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders**|NeurIPS2023|[CROMA](https://arxiv.org/pdf/2311.00566.pdf)|[link](https://github.com/antofuller/CROMA)|
|**Cross-Scale MAE**|**Cross-Scale MAE: A Tale of Multiscale Exploitation in Remote Sensing**|NeurIPS2023|[Cross-Scale MAE](https://openreview.net/pdf?id=5oEVdOd6TV)|[link](https://github.com/aicip/Cross-Scale-MAE)|
|**DeCUR**|**DeCUR: decoupling common & unique representations for multimodal self-supervision**|Arxiv2023|[DeCUR](https://arxiv.org/abs/2309.05300)|[link](https://github.com/zhu-xlab/DeCUR)|
|**Presto**|**Lightweight, Pre-trained Transformers for Remote Sensing Timeseries**|Arxiv2023|[Presto](https://arxiv.org/abs/2304.14065)|[link](https://github.com/nasaharvest/presto)|
|**CtxMIM**|**CtxMIM: Context-Enhanced Masked Image Modeling for Remote Sensing Image Understanding**|Arxiv2023|[CtxMIM](https://arxiv.org/abs/2310.00022)|null|
|**FG-MAE**|**Feature Guided Masked Autoencoder for Self-supervised Learning in Remote Sensing**|Arxiv2023|[FG-MAE](https://arxiv.org/abs/2310.18653)|[link](https://github.com/zhu-xlab/FGMAE)|
|**Prithvi**|**Foundation Models for Generalist Geospatial Artificial Intelligence**|Arxiv2023|[Prithvi](https://arxiv.org/abs/2310.18660)|[link](https://huggingface.co/ibm-nasa-geospatial)|
|**RingMo-lite**|**RingMo-lite: A Remote Sensing Multi-task Lightweight Network with CNN-Transformer Hybrid Framework**|Arxiv2023|[RingMo-lite](https://arxiv.org/abs/2309.09003)|null|
|**-**|**A Self-Supervised Cross-Modal Remote Sensing Foundation Model with Multi-Domain Representation and Cross-Domain Fusion**|IGARSS2023|[Paper](https://ieeexplore.ieee.org/abstract/document/10282433)|null|
|**EarthPT**|**EarthPT: a foundation model for Earth Observation**|NeurIPS2023 CCAI workshop|[EarthPT](https://arxiv.org/abs/2309.07207)|[link](https://github.com/aspiaspace/EarthPT)|
|**USat**|**USat: A Unified Self-Supervised Encoder for Multi-Sensor Satellite Imagery**|Arxiv2023|[USat](https://arxiv.org/abs/2312.02199)|[link](https://github.com/stanfordmlgroup/USat)|
|**FoMo-Bench**|**FoMo-Bench: a multi-modal, multi-scale and multi-task Forest Monitoring Benchmark for remote sensing foundation models**|Arxiv2023|[FoMo-Bench](https://arxiv.org/abs/2312.10114)|[link](https://github.com/RolnickLab/FoMo-Bench)|
|**AIEarth**|**Analytical Insight of Earth: A Cloud-Platform of Intelligent Computing for Geospatial Big Data**|Arxiv2023|[AIEarth](https://arxiv.org/abs/2312.16385)|[link](https://engine-aiearth.aliyun.com/#/)|
|**-**|**Self-Supervised Learning for SAR ATR with a Knowledge-Guided Predictive Architecture**|Arxiv2023|[Paper](https://arxiv.org/abs/2311.15153v4)|[link](https://github.com/waterdisappear/SAR-JEPA)|
|**Clay**|**Clay Foundation Model**|-|null|[link](https://clay-foundation.github.io/model/)|
|**Hydro**|****Hydro--A Foundation Model for Water in Satellite Imagery****|-|null|[link](https://github.com/isaaccorley/hydro-foundation-model)|
|**U-BARN**|**Self-Supervised Spatio-Temporal Representation Learning of Satellite Image Time Series**|JSTARS2024|[Paper](https://ieeexplore.ieee.org/document/10414422)|[link](https://src.koda.cnrs.fr/iris.dumeur/ssl_ubarn)|
|**GeRSP**|**Generic Knowledge Boosted Pre-training For Remote Sensing Images**|Arxiv2024|[GeRSP](https://arxiv.org/abs/2401.04614)|[GeRSP](https://github.com/floatingstarZ/GeRSP)|
|**SwiMDiff**|**SwiMDiff: Scene-wide Matching Contrastive Learning with Diffusion Constraint for Remote Sensing Image**|Arxiv2024|[SwiMDiff](https://arxiv.org/abs/2401.05093)|null|
|**OFA-Net**|**One for All: Toward Unified Foundation Models for Earth Vision**|Arxiv2024|[OFA-Net](https://arxiv.org/abs/2401.07527)|null|
|**SMLFR**|**Generative ConvNet Foundation Model With Sparse Modeling and Low-Frequency Reconstruction for Remote Sensing Image Interpretation**|TGRS2024|[SMLFR](https://ieeexplore.ieee.org/abstract/document/10378718)|[link](https://github.com/HIT-SIRS/SMLFR)|
|**SpectralGPT**|**SpectralGPT: Spectral Foundation Model**|TPAMI2024|[SpectralGPT](https://arxiv.org/abs/2311.07113)|[link](https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT)|
|**S2MAE**|**S2MAE: A Spatial-Spectral Pretraining Foundation Model for Spectral Remote Sensing Data**|CVPR2024|[S2MAE](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_S2MAE_A_Spatial-Spectral_Pretraining_Foundation_Model_for_Spectral_Remote_Sensing_CVPR_2024_paper.pdf)|null|
|**SatMAE++**|**Rethinking Transformers Pre-training for Multi-Spectral Satellite Imagery**|CVPR2024|[SatMAE++](https://arxiv.org/abs/2403.05419)|[link](https://github.com/techmn/satmae_pp)|
|**msGFM**|**Bridging Remote Sensors with Multisensor Geospatial Foundation Models**|CVPR2024|[msGFM](https://arxiv.org/abs/2404.01260)|[link](https://github.com/boranhan/Geospatial_Foundation_Models)|
|**SkySense**|**SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery**|CVPR2024|[SkySense](https://arxiv.org/abs/2312.10115)|Comming soon|
|**MTP**|**MTP: Advancing Remote Sensing Foundation Model via Multi-Task Pretraining**|Arxiv2024|[MTP](https://arxiv.org/abs/2403.13430)|[link](https://github.com/ViTAE-Transformer/MTP)|
|**DOFA**|**Neural Plasticity-Inspired Foundation Model for Observing the Earth Crossing Modalities**|Arxiv2024|[DOFA](https://arxiv.org/abs/2403.15356)|[link](https://github.com/zhu-xlab/DOFA)|
|**PIS**|**Pretrain A Remote Sensing Foundation Model by Promoting Intra-instance Similarity**|-|null|[link](https://github.com/ShawnAn-WHU/PIS)|
|**MMEarth**|**MMEarth: Exploring Multi-Modal Pretext Tasks For Geospatial Representation Learning**|Arxiv2024|[MMEarth](https://arxiv.org/abs/2405.02771)|[link](https://vishalned.github.io/mmearth/)|
|**SARATR-X**|**SARATR-X: A Foundation Model for Synthetic Aperture Radar Images Target Recognition**|Arxiv2024|[SARATR-X](https://arxiv.org/abs/2405.09365)|[link](https://github.com/waterdisappear/SARATR-X)|
|**LeMeViT**|**LeMeViT: Efficient Vision Transformer with Learnable Meta Tokens for Remote Sensing Image Interpretation**|IJCAI2024|[LeMeViT](https://arxiv.org/abs/2405.09789)|[link](https://github.com/ViTAE-Transformer/LeMeViT/tree/main?tab=readme-ov-file)|
|**SoftCon**|**Multi-Label Guided Soft Contrastive Learning for Efficient Earth Observation Pretraining**|Arxiv2024|[SoftCon](https://arxiv.org/abs/2405.20462)|[link](https://github.com/zhu-xlab/softcon?tab=readme-ov-file)|
|**RS-DFM**|**RS-DFM: A Remote Sensing Distributed Foundation Model for Diverse Downstream Tasks**|Arxiv2024|[RS-DFM](https://arxiv.org/abs/2406.07032)|null|
|**A2-MAE**|**A2-MAE: A spatial-temporal-spectral unified remote sensing pre-training method based on anchor-aware masked autoencoder**|Arxiv2024|[A2-MAE](https://arxiv.org/abs/2406.08079)|null|
|**HyperSIGMA**|**HyperSIGMA: Hyperspectral Intelligence Comprehension Foundation Model**|Arxiv2024|[HyperSIGMA](https://arxiv.org/abs/2406.11519)|[link](https://github.com/WHU-Sigma/HyperSIGMA?tab=readme-ov-file)|
|**SelectiveMAE**|**Scaling Efficient Masked Autoencoder Learning on Large Remote Sensing Dataset**|Arxiv2024|[SelectiveMAE](https://arxiv.org/abs/2406.11933)|[link](https://github.com/Fengxiang23/SelectiveMAE)|
|**OmniSat**|**OmniSat: Self-Supervised Modality Fusion for Earth Observation**|ECCV2024|[OmniSat](https://arxiv.org/pdf/2404.08351)|[link](https://github.com/gastruc/OmniSat?tab=readme-ov-file)|
|**MM-VSF**|**Towards a Knowledge guided Multimodal Foundation Model for Spatio-Temporal Remote Sensing Applications**|Arxiv2024|[MM-VSF](https://arxiv.org/pdf/2407.19660)|null|
|**MA3E**|**Masked Angle-Aware Autoencoder for Remote Sensing Images**|ECCV2024|[MA3E](https://arxiv.org/abs/2408.01946)|[link](https://github.com/benesakitam/MA3E)|
|**SpectralEarth**|**SpectralEarth: Training Hyperspectral Foundation Models at Scale**|Arxiv2024|[SpectralEarth](https://arxiv.org/abs/2408.08447)|null|

## Remote Sensing <ins>Vision-Language</ins> Foundation Models

|Abbreviation|Title|Publication|Paper|Code & Weights|
|:---:|---|:---:|:---:|:---:|
|**RSGPT**|**RSGPT: A Remote Sensing Vision Language Model and Benchmark**|Arxiv2023|[RSGPT](https://arxiv.org/abs/2307.15266)|[link](https://github.com/Lavender105/RSGPT)|
|**RemoteCLIP**|**RemoteCLIP: A Vision Language Foundation Model for Remote Sensing**|Arxiv2023|[RemoteCLIP](https://arxiv.org/abs/2306.11029)|[link](https://github.com/ChenDelong1999/RemoteCLIP)|
|**GeoRSCLIP**|**RS5M: A Large Scale Vision-Language Dataset for Remote Sensing Vision-Language Foundation Model**|Arxiv2023|[GeoRSCLIP](https://arxiv.org/abs/2306.11300)|[link](https://github.com/om-ai-lab/RS5M?tab=readme-ov-file)|
|**GRAFT**|**Remote Sensing Vision-Language Foundation Models without Annotations via Ground Remote Alignment**|ICLR2024|[GRAFT](https://openreview.net/pdf?id=w9tc699w3Z)|null|
|**-**|**Charting New Territories: Exploring the Geographic and Geospatial Capabilities of Multimodal LLMs**|Arxiv2023|[Paper](https://arxiv.org/abs/2311.14656)|[link](https://github.com/jonathan-roberts1/charting-new-territories)|
|**-**|**Remote Sensing ChatGPT: Solving Remote Sensing Tasks with ChatGPT and Visual Models**|Arxiv2024|[Paper](https://arxiv.org/abs/2401.09083)|[link](https://github.com/HaonanGuo/Remote-Sensing-ChatGPT)|
|**SkyEyeGPT**|**SkyEyeGPT: Unifying Remote Sensing Vision-Language Tasks via Instruction Tuning with Large Language Model**|Arxiv2024|[Paper](https://arxiv.org/abs/2401.09712)|[link](https://github.com/ZhanYang-nwpu/SkyEyeGPT)|
|**EarthGPT**|**EarthGPT: A Universal Multi-modal Large Language Model for Multi-sensor Image Comprehension in Remote Sensing Domain**|Arxiv2024|[Paper](https://arxiv.org/abs/2401.16822)|null|
|**SkyCLIP**|**SkyScript: A Large and Semantically Diverse Vision-Language Dataset for Remote Sensing**|AAAI2024|[SkyCLIP](https://arxiv.org/abs/2312.12856)|[link](https://github.com/wangzhecheng/SkyScript)|
|**GeoChat**|**GeoChat: Grounded Large Vision-Language Model for Remote Sensing**|CVPR2024|[GeoChat](https://arxiv.org/abs/2311.15826)|[link](https://github.com/mbzuai-oryx/GeoChat)|
|**LHRS-Bot**|**LHRS-Bot: Empowering Remote Sensing with VGI-Enhanced Large Multimodal Language Model**|Arxiv2024|[Paper](https://arxiv.org/abs/2402.02544)|[link](https://github.com/NJU-LHRS/LHRS-Bot)|
|**H2RSVLM**|**H2RSVLM: Towards Helpful and Honest Remote Sensing Large Vision Language Model**|Arxiv2024|[Paper](https://arxiv.org/abs/2403.20213)|[link](https://github.com/opendatalab/H2RSVLM)|
|**RS-LLaVA**|**RS-LLaVA: Large Vision Language Model for Joint Captioning and Question Answering in Remote Sensing Imagery**|RS2024|[Paper](https://www.mdpi.com/2072-4292/16/9/1477)|[link](https://github.com/BigData-KSU/RS-LLaVA?tab=readme-ov-file)|
|**SkySenseGPT**|**SkySenseGPT: A Fine-Grained Instruction Tuning Dataset and Model for Remote Sensing Vision-Language Understanding**|Arxiv2024|[Paper]()|[link](https://github.com/Luo-Z13/SkySenseGPT)|


## Remote Sensing <ins>Generative</ins> Foundation Models

|Abbreviation|Title|Publication|Paper|Code & Weights|
|:---:|---|:---:|:---:|:---:|
|**Seg2Sat**|**Seg2Sat - Segmentation to aerial view using pretrained diffuser models**|Github|null|[link](https://github.com/RubenGres/Seg2Sat)|
|**-**|**Generate Your Own Scotland: Satellite Image Generation Conditioned on Maps**|NeurIPSW2023|[Paper](https://arxiv.org/abs/2308.16648)|[link](https://github.com/toastyfrosty/map-sat)|
|**GeoRSSD**|**RS5M: A Large Scale Vision-Language Dataset for Remote Sensing Vision-Language Foundation Model**|Arxiv2023|[Paper](https://arxiv.org/abs/2306.11300)|[link](https://huggingface.co/Zilun/GeoRSSD)|
|**DiffusionSat**|**DiffusionSat: A Generative Foundation Model for Satellite Imagery**|ICLR2024|[DiffusionSat](https://arxiv.org/abs/2312.03606)|[link](https://github.com/samar-khanna/DiffusionSat)|
|**CRS-Diff**|**CRS-Diff: Controllable Generative Remote Sensing Foundation Model**|Arxiv2024|[Paper](https://arxiv.org/abs/2403.11614)|null|
|**MetaEarth**|**MetaEarth: A Generative Foundation Model for Global-Scale Remote Sensing Image Generation**|Arxiv2024|[Paper](https://arxiv.org/abs/2405.13570)|[link](https://jiupinjia.github.io/metaearth/)|



## Remote Sensing <ins>Vision-Location</ins> Foundation Models

|Abbreviation|Title|Publication|Paper|Code & Weights|
|:---:|---|:---:|:---:|:---:|
|**CSP**|**CSP: Self-Supervised Contrastive Spatial Pre-Training for Geospatial-Visual Representations**|ICML2023|[CSP](https://arxiv.org/abs/2305.01118)|[link](https://gengchenmai.github.io/csp-website/)|
|**GeoCLIP**|**GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization**|NeurIPS2023|[GeoCLIP](https://arxiv.org/abs/2309.16020)|[link](https://vicentevivan.github.io/GeoCLIP/)|
|**SatCLIP**|**SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery**|Arxiv2023|[SatCLIP](https://arxiv.org/abs/2311.17179)|[link](https://github.com/microsoft/satclip)|

## Remote Sensing <ins>Vision-Audio</ins> Foundation Models

|Abbreviation|Title|Publication|Paper|Code & Weights|
|:---:|---|:---:|:---:|:---:|
|**-**|**Self-supervised audiovisual representation learning for remote sensing data**|JAG2022|[Paper](https://www.sciencedirect.com/science/article/pii/S1569843222003181)|[link](https://github.com/khdlr/SoundingEarth)|


## Remote Sensing <ins>Task-specific</ins> Foundation Models

|Abbreviation|Title|Publication|Paper|Code & Weights|Task|
|:---:|---|:---:|:---:|:---:|:---:|
|**SS-MAE**|**SS-MAE: Spatial-Spectral Masked Auto-Encoder for Mulit-Source Remote Sensing Image Classification**|TGRS2023|[Paper](https://ieeexplore.ieee.org/document/10314566/)|[link](https://github.com/summitgao/SS-MAE?tab=readme-ov-file)|Image Classification|
|**TTP**|**Time Travelling Pixels: Bitemporal Features Integration with Foundation Model for Remote Sensing Image Change Detection**|Arxiv2023|[Paper](https://arxiv.org/abs/2312.16202)|[link](https://github.com/KyanChen/TTP)|Change Detection|
|**CSMAE**|**Exploring Masked Autoencoders for Sensor-Agnostic Image Retrieval in Remote Sensing**|Arxiv2024|[Paper](https://arxiv.org/abs/2401.07782)|[link](https://github.com/jakhac/CSMAE)|Image Retrieval|
|**RSPrompter**|**RSPrompter: Learning to Prompt for Remote Sensing Instance Segmentation based on Visual Foundation Model**|TGRS2024|[Paper](https://arxiv.org/abs/2306.16269)|[link](https://github.com/KyanChen/RSPrompter)|Instance Segmentation|
|**BAN**|**A New Learning Paradigm for Foundation Model-based Remote Sensing Change Detection**|TGRS2024|[Paper](https://arxiv.org/abs/2312.01163)|[link](https://github.com/likyoo/BAN)|Change Detection|
|**-**|**Change Detection Between Optical Remote Sensing Imagery and Map Data via Segment Anything Model (SAM)**|Arxiv2024|[Paper](https://arxiv.org/abs/2401.09019)|null|Change Detection (Optical & OSM data)|
|**AnyChange**|**Segment Any Change**|Arxiv2024|[Paper](https://arxiv.org/abs/2402.01188)|null|Zero-shot Change Detection|
|**RS-CapRet**|**Large Language Models for Captioning and Retrieving Remote Sensing Images**|Arxiv2024|[Paper](https://arxiv.org/abs/2402.06475)|null|Image Caption & Text-image Retrieval|
|**-**|**Task Specific Pretraining with Noisy Labels for Remote sensing Image Segmentation**|Arxiv2024|[Paper](https://arxiv.org/abs/2402.16164)|null|Image Segmentation (Noisy labels)|
|**RSBuilding**|**RSBuilding: Towards General Remote Sensing Image Building Extraction and Change Detection with Foundation Model**|Arxiv2024|[Paper](https://arxiv.org/abs/2403.07564)|[link](https://github.com/Meize0729/RSBuilding)|Building Extraction and Change Detection|
|**SAM-Road**|**Segment Anything Model for Road Network Graph Extraction**|Arxiv2024|[Paper](https://arxiv.org/abs/2403.16051)|[link](https://github.com/htcr/sam_road)|Road Extraction|

## Remote Sensing Agents
|Abbreviation|Title|Publication|Paper|Code & Weights|
|:---:|---|:---:|:---:|:---:|
|**GeoLLM-QA**|**Evaluating Tool-Augmented Agents in Remote Sensing Platforms**|ICLR 2024 ML4RS Workshop|[Paper](https://arxiv.org/abs/2405.00709)|null|
|**RS-Agent**|**RS-Agent: Automating Remote Sensing Tasks through Intelligent Agents**|Arxiv2024|[Paper](https://arxiv.org/abs/2406.07089)|null|


## Benchmarks for RSFMs
|Abbreviation|Title|Publication|Paper|Link|Downstream Tasks|
|:---:|---|:---:|:---:|:---:|:---:|
|**-**|**Revisiting pre-trained remote sensing model benchmarks: resizing and normalization matters**|Arxiv2023|[Paper](https://arxiv.org/abs/2305.13456)|[link](https://github.com/isaaccorley/resize-is-all-you-need)|Classification|
|**GEO-Bench**|**GEO-Bench: Toward Foundation Models for Earth Monitoring**|Arxiv2023|[Paper](https://arxiv.org/abs/2306.03831)|[link](https://github.com/ServiceNow/geo-bench)|Classification & Segmentation|
|**FoMo-Bench**|**FoMo-Bench: a multi-modal, multi-scale and multi-task Forest Monitoring Benchmark for remote sensing foundation models**|Arxiv2023|[FoMo-Bench](https://arxiv.org/abs/2312.10114)|Comming soon|Classification & Segmentation & Detection for forest monitoring|
|**PhilEO**|**PhilEO Bench: Evaluating Geo-Spatial Foundation Models**|Arxiv2024|[Paper](https://arxiv.org/abs/2401.04464)|[link](https://github.com/91097luke/phileo-bench)|Segmentation & Regression estimation|
|**SkySense**|**SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery**|CVPR2024|[SkySense](https://arxiv.org/abs/2312.10115)|Comming Soon|Classification & Segmentation & Detection & Change detection & Multi-Modal Segmentation: Time-insensitive LandCover Mapping & Multi-Modal Segmentation: Time-sensitive Crop Mapping & Multi-Modal Scene Classification|
|**VLEO-Bench**|**Good at captioning, bad at counting: Benchmarking GPT-4V on Earth observation data**|Arxiv2024|[VLEO-bench](https://arxiv.org/abs/2401.17600)|[link](https://vleo.danielz.ch/)| Location Recognition & Captioning & Scene Classification & Counting & Detection & Change detection|
|**VRSBench**|**VRSBench: A Versatile Vision-Language Benchmark Dataset for Remote Sensing Image Understanding**|Arxiv2024|[VRSBench](https://arxiv.org/abs/2406.12384)|[link](https://vrsbench.github.io/)|Image Captioning & Object Referring & Visual Question Answering|


## (Large-scale) Pre-training Datasets

|Abbreviation|Title|Publication|Paper|Attribute|Link|
|:---:|---|:---:|:---:|:---:|:---:|
|**fMoW**|**Functional Map of the World**|CVPR2018|[fMoW](https://openaccess.thecvf.com/content_cvpr_2018/html/Christie_Functional_Map_of_CVPR_2018_paper.html)|**Vision**|[link](https://github.com/fMoW)|
|**SEN12MS**|**SEN12MS -- A Curated Dataset of Georeferenced Multi-Spectral Sentinel-1/2 Imagery for Deep Learning and Data Fusion**|-|[SEN12MS](https://arxiv.org/abs/1906.07789)|**Vision**|[link](https://arxiv.org/abs/1906.07789)|
|**BEN-MM**|**BigEarthNet-MM: A Large Scale Multi-Modal Multi-Label Benchmark Archive for Remote Sensing Image Classification and Retrieval**|GRSM2021|[BEN-MM](https://ieeexplore.ieee.org/abstract/document/9552024)|**Vision**|[link](https://ieeexplore.ieee.org/abstract/document/9552024)|
|**MillionAID**|**On Creating Benchmark Dataset for Aerial Image Interpretation: Reviews, Guidances, and Million-AID**|JSTARS2021|[MillionAID](https://ieeexplore.ieee.org/abstract/document/9393553)|**Vision**|[link](https://captain-whu.github.io/DiRS/)|
|**SeCo**|**Seasonal Contrast: Unsupervised Pre-Training From Uncurated Remote Sensing Data**|ICCV2021|[SeCo](https://openaccess.thecvf.com/content/ICCV2021/html/Manas_Seasonal_Contrast_Unsupervised_Pre-Training_From_Uncurated_Remote_Sensing_Data_ICCV_2021_paper.html)|**Vision**|[link](https://github.com/ServiceNow/seasonal-contrast)|
|**fMoW-S2**|**SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery**|NeurIPS2022|[fMoW-S2](https://proceedings.neurips.cc/paper_files/paper/2022/hash/01c561df365429f33fcd7a7faa44c985-Abstract-Conference.html)|**Vision**|[link](https://purl.stanford.edu/vg497cb6002)|
|**TOV-RS-Balanced**|**TOV: The original vision model for optical remote sensing image understanding via self-supervised learning**|JSTARS2023|[TOV](https://ieeexplore.ieee.org/abstract/document/10110958)|**Vision**|[link](https://github.com/GeoX-Lab/G-RSIM/tree/main/TOV_v1)|
|**SSL4EO-S12**|**SSL4EO-S12: A Large-Scale Multi-Modal, Multi-Temporal Dataset for Self-Supervised Learning in Earth Observation**|GRSM2023|[SSL4EO-S12](https://arxiv.org/abs/2211.07044)|**Vision**|[link](https://github.com/zhu-xlab/SSL4EO-S12)|
|**SSL4EO-L**|**SSL4EO-L: Datasets and Foundation Models for Landsat Imagery**|Arxiv2023|[SSL4EO-L](https://arxiv.org/abs/2306.09424)|**Vision**|[link](https://github.com/microsoft/torchgeo)|
|**SatlasPretrain**|**SatlasPretrain: A Large-Scale Dataset for Remote Sensing Image Understanding**|ICCV2023|[SatlasPretrain](https://arxiv.org/abs/2211.15660)|**Vision (Supervised)**|[link](https://github.com/allenai/satlas/blob/main/SatlasPretrain.md)|
|**CACo**|**Change-Aware Sampling and Contrastive Learning for Satellite Images**|CVPR2023|[CACo](https://openaccess.thecvf.com/content/CVPR2023/html/Mall_Change-Aware_Sampling_and_Contrastive_Learning_for_Satellite_Images_CVPR_2023_paper.html)|**Vision**|[Comming soon](https://github.com/utkarshmall13/CACo)|
|**SAMRS**|**SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model**|NeurIPS2023|[SAMRS](https://arxiv.org/abs/2305.02034)|**Vision**|[link](https://github.com/ViTAE-Transformer/SAMRS)|
|**RSVG**|**RSVG: Exploring Data and Models for Visual Grounding on Remote Sensing Data**|TGRS2023|[RSVG](https://ieeexplore.ieee.org/document/10056343)|**Vision-Language**|[link](https://github.com/ZhanYang-nwpu/RSVG-pytorch)|
|**RS5M**|**RS5M: A Large Scale Vision-Language Dataset for Remote Sensing Vision-Language Foundation Model**|Arxiv2023|[RS5M](https://arxiv.org/abs/2306.11300)|**Vision-Language**|[link](https://github.com/om-ai-lab/RS5M)|
|**GEO-Bench**|**GEO-Bench: Toward Foundation Models for Earth Monitoring**|Arxiv2023|[GEO-Bench](https://arxiv.org/abs/2306.03831)|**Vision (Evaluation)**|[link](https://github.com/ServiceNow/geo-bench)|
|**RSICap & RSIEval**|**RSGPT: A Remote Sensing Vision Language Model and Benchmark**|Arxiv2023|[RSGPT](https://arxiv.org/abs/2307.15266)|**Vision-Language**|[Comming soon](https://github.com/Lavender105/RSGPT)|
|**Clay**|**Clay Foundation Model**|-|null|**Vision**|[link](https://clay-foundation.github.io/model/)|
|**SATIN**|**SATIN: A Multi-Task Metadataset for Classifying Satellite Imagery using Vision-Language Models**|ICCVW2023|[SATIN](https://arxiv.org/abs/2304.11619)|**Vision-Language**|[link](https://satinbenchmark.github.io/)|
|**SkyScript**|**SkyScript: A Large and Semantically Diverse Vision-Language Dataset for Remote Sensing**|AAAI2024|[SkyScript](https://arxiv.org/abs/2312.12856)|**Vision-Language**|[link](https://github.com/wangzhecheng/SkyScript)|
|**ChatEarthNet**|**ChatEarthNet: A Global-Scale, High-Quality Image-Text Dataset for Remote Sensing**|Arxiv2024|[ChatEarthNet](https://arxiv.org/abs/2402.11325)|**Vision-Language**|[link](https://github.com/zhu-xlab/ChatEarthNet)|
|**LuoJiaHOG**|**LuoJiaHOG: A Hierarchy Oriented Geo-aware Image Caption Dataset for Remote Sensing Image-Text Retrieval**|Arxiv2024|[LuoJiaHOG](https://arxiv.org/abs/2403.10887)|**Vision-Language**|null|
|**MMEarth**|**MMEarth: Exploring Multi-Modal Pretext Tasks For Geospatial Representation Learning**|Arxiv2024|[MMEarth](https://arxiv.org/abs/2405.02771)|**Vision**|[link](https://vishalned.github.io/mmearth/)|
|**SeeFar**|**SeeFar: Satellite Agnostic Multi-Resolution Dataset for Geospatial Foundation Models**|Arxiv2024|[SeeFar](https://arxiv.org/abs/2406.06776)|**Vision**|[link](https://coastalcarbon.ai/seefar)|
|**FIT-RS**|**SkySenseGPT: A Fine-Grained Instruction Tuning Dataset and Model for Remote Sensing Vision-Language Understanding**|Arxiv2024|[Paper]()|**Vision-Language**|[link](https://github.com/Luo-Z13/SkySenseGPT)|
|**RS-GPT4V**|**RS-GPT4V: A Unified Multimodal Instruction-Following Dataset for Remote Sensing Image Understanding**|Arxiv2024|[Paper](https://arxiv.org/abs/2406.12479)|**Vision-Language**|[link](https://github.com/GeoX-Lab/RS-GPT4V/tree/main)|
|**RS-4M**|**Scaling Efficient Masked Autoencoder Learning on Large Remote Sensing Dataset**|Arxiv2024|[RS-4M](https://arxiv.org/abs/2406.11933)|**Vision**|[link](https://github.com/Fengxiang23/SelectiveMAE)|
|**Major TOM**|**Major TOM: Expandable Datasets for Earth Observation**|Arxiv2024|[Major TOM](https://arxiv.org/abs/2402.12095)|**Vision**|[link](https://huggingface.co/Major-TOM)|
|**VRSBench**|**VRSBench: A Versatile Vision-Language Benchmark Dataset for Remote Sensing Image Understanding**|Arxiv2024|[VRSBench](https://arxiv.org/abs/2406.12384)|**Vision-Language**|[link](https://vrsbench.github.io/)|

# Relevant Projects
*（TODO. This section is dedicated to recommending more relevant and impactful projects, with the hope of promoting the development of the RS community. :smile: :rocket:）*
|Title|Link|Brief Introduction|
|---|:---:|:---:|
|**RSFMs (Remote Sensing Foundation Models) Playground**|[link](https://github.com/synativ/RSFMs)|An open-source playground to streamline the evaluation and fine-tuning of RSFMs on various datasets.|

## Survey Papers
|Title|Publication|Paper|Attribute|
|---|:---:|:---:|:---:|
|**Self-Supervised Remote Sensing Feature Learning: Learning Paradigms, Challenges, and Future Works**|TGRS2023|[Paper](https://ieeexplore.ieee.org/abstract/document/10126079)|**Vision & Vision-Language**|
|**The Potential of Visual ChatGPT For Remote Sensing**|Arxiv2023|[Paper](https://arxiv.org/abs/2304.13009)|**Vision-Language**|
|**遥感大模型：进展与前瞻**|武汉大学学报 (信息科学版) 2023|[Paper](http://ch.whu.edu.cn/cn/article/doi/10.13203/j.whugis20230341?viewType=HTML)|**Vision & Vision-Language**|
|**地理人工智能样本：模型、质量与服务**|武汉大学学报 (信息科学版) 2023|[Paper](http://ch.whu.edu.cn/article/id/5e67ed6a-aae5-4ec0-ad1b-f2aba89f4617)|**-**|
|**Brain-Inspired Remote Sensing Foundation Models and Open Problems: A Comprehensive Survey**|JSTARS2023|[Paper](https://ieeexplore.ieee.org/abstract/document/10254282)|**Vision & Vision-Language**|
|**Revisiting pre-trained remote sensing model benchmarks: resizing and normalization matters**|Arxiv2023|[Paper](https://arxiv.org/abs/2305.13456)|**Vision**|
|**An Agenda for Multimodal Foundation Models for Earth Observation**|IGARSS2023|[Paper](https://ieeexplore.ieee.org/abstract/document/10282966)|**Vision**|
|**Transfer learning in environmental remote sensing**|RSE2024|[Paper](https://www.sciencedirect.com/science/article/pii/S0034425723004765)|**Transfer learning**|
|**遥感基础模型发展综述与未来设想**|遥感学报2023|[Paper](https://www.ygxb.ac.cn/zh/article/doi/10.11834/jrs.20233313/)|**-**|
|**On the Promises and Challenges of Multimodal Foundation Models for Geographical, Environmental, Agricultural, and Urban Planning Applications**|Arxiv2023|[Paper](https://arxiv.org/abs/2312.17016)|**Vision-Language**|
|**Vision-Language Models in Remote Sensing: Current Progress and Future Trends**|IEEE GRSM2024|[Paper](https://arxiv.org/abs/2305.05726)|**Vision-Language**|
|**On the Foundations of Earth and Climate Foundation Models**|Arxiv2024|[Paper](https://arxiv.org/abs/2405.04285)|**Vision & Vision-Language**|
|**Towards Vision-Language Geo-Foundation Model: A Survey**|Arxiv2024|[Paper](https://arxiv.org/abs/2406.09385)|**Vision-Language**|
|**AI Foundation Models in Remote Sensing: A Survey**|Arxiv2024|[Paper](https://arxiv.org/abs/2408.03464)|**Vision**|

## Citation

If you find this repository useful, please consider giving a star :star: and citation:

```
@inproceedings{guo2024skysense,
  title={Skysense: A multi-modal remote sensing foundation model towards universal interpretation for earth observation imagery},
  author={Guo, Xin and Lao, Jiangwei and Dang, Bo and Zhang, Yingying and Yu, Lei and Ru, Lixiang and Zhong, Liheng and Huang, Ziyuan and Wu, Kang and Hu, Dingxiang and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={27672--27683},
  year={2024}
}
```
