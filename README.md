[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models/graphs/commit-activity)
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models)
<img alt="GitHub watchers" src="https://img.shields.io/github/watchers/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models?style=social"> <img alt="GitHub stars" src="https://img.shields.io/github/stars/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models?style=social"> <img alt="GitHub forks" src="https://img.shields.io/github/forks/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models?style=social">

# <p align=center>`Awesome Remote Sensing Foundation Models`</p>

:star2:**A collection of papers, datasets, benchmarks, code, and pre-trained weights for Remote Sensing Foundation Models (RSFMs).**

## 📢 Latest Updates
:fire::fire::fire: Last Updated on 2026.05.06 :fire::fire::fire:

## Table of Contents
- **Models**
  - [Remote Sensing Vision Foundation Models](#remote-sensing-vision-foundation-models)
  - [Remote Sensing Vision-Language Foundation Models](#remote-sensing-vision-language-foundation-models)
  - [Remote Sensing Generative Foundation Models](#remote-sensing-generative-foundation-models)
  - [Remote Sensing Vision-Location Foundation Models](#remote-sensing-vision-location-foundation-models)
  - [Remote Sensing Vision-Audio Foundation Models](#remote-sensing-vision-audio-foundation-models)
  - [Remote Sensing Agents](#remote-sensing-agents)
- **Datasets & Benchmarks**
  - [Benchmarks for RSFMs](#benchmarks-for-rsfms)
  - [(Large-scale) Pre-training Datasets](#large-scale-pre-training-datasets)
  - [Embeddings data](#embeddings-data)
- **Others**
  - [Relevant Projects](#relevant-projects)
  - [Survey/Commentary Papers](#surveycommentary-papers)

## Remote Sensing <ins>Vision</ins> Foundation Models
|Abbreviation|Title|Publication|Paper|Code & Weights|
|:---:|---|:---:|:---:|:---:|
|**GeoKR**|**Geographical Knowledge-Driven Representation Learning for Remote Sensing Images**|TGRS2021|[GeoKR](https://ieeexplore.ieee.org/abstract/document/9559903)|[link](https://github.com/flyakon/Geographical-Knowledge-driven-Representaion-Learning)|
|**-**|**Self-Supervised Learning of Remote Sensing Scene Representations Using Contrastive Multiview Coding**|CVPRW2021|[Paper](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/html/Stojnic_Self-Supervised_Learning_of_Remote_Sensing_Scene_Representations_Using_Contrastive_Multiview_CVPRW_2021_paper.html)|[link](https://github.com/vladan-stojnic/CMC-RSSR)|
|**GASSL**|**Geography-Aware Self-Supervised Learning**|ICCV2021|[GASSL](https://openaccess.thecvf.com/content/ICCV2021/html/Ayush_Geography-Aware_Self-Supervised_Learning_ICCV_2021_paper.html)|[link](https://github.com/sustainlab-group/geography-aware-ssl)|
|**SeCo**|**Seasonal Contrast: Unsupervised Pre-Training From Uncurated Remote Sensing Data**|ICCV2021|[SeCo](https://openaccess.thecvf.com/content/ICCV2021/html/Manas_Seasonal_Contrast_Unsupervised_Pre-Training_From_Uncurated_Remote_Sensing_Data_ICCV_2021_paper.html)|[link](https://github.com/ServiceNow/seasonal-contrast)|
|**RSP**|**An Empirical Study of Remote Sensing Pretraining**|TGRS2022|[RSP](https://ieeexplore.ieee.org/abstract/document/9782149)|[link](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA)|
|**MATTER**|**Self-Supervised Material and Texture Representation Learning for Remote Sensing Tasks**|CVPR2022|[MATTER](https://openaccess.thecvf.com/content/CVPR2022/html/Akiva_Self-Supervised_Material_and_Texture_Representation_Learning_for_Remote_Sensing_Tasks_CVPR_2022_paper.html)|[link](https://github.com/periakiva/MATTER)|
|**-**|**Self-supervised Vision Transformers for Land-cover Segmentation and Classification**|CVPRW2022|[Paper](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/html/Scheibenreif_Self-Supervised_Vision_Transformers_for_Land-Cover_Segmentation_and_Classification_CVPRW_2022_paper.html)|[link](https://github.com/HSG-AIML/SSLTransformerRS)|
|**DINO-MM**|**Self-Supervised Vision Transformers for Joint SAR-Optical Representation Learning**|IGARSS2022|[DINO-MM](https://doi.org/10.1109/igarss46834.2022.9883983)|[link](https://github.com/zhu-xlab/DINO-MM)|
|**GeCo**|**Geographical Supervision Correction for Remote Sensing Representation Learning**|TGRS2022|[GeCo](https://ieeexplore.ieee.org/abstract/document/9869651)|[link](https://github.com/GeoX-Lab/G-RSIM)|
|**RingMo**|**RingMo: A remote sensing foundation model with masked image modeling**|TGRS2022|[RingMo](https://ieeexplore.ieee.org/abstract/document/9844015)|[Code](https://github.com/comeony/RingMo)|
|**RS-BYOL**|**Self-Supervised Learning for Invariant Representations From Multi-Spectral and SAR Images**|JSTARS2022|[RS-BYOL](https://ieeexplore.ieee.org/abstract/document/9880533)|[link](https://ieeexplore.ieee.org/abstract/document/9880533)|
|**CSPT**|**Consecutive Pre-Training: A Knowledge Transfer Learning Strategy with Relevant Unlabeled Data for Remote Sensing Domain**|RS2022|[CSPT](https://www.mdpi.com/2072-4292/14/22/5675#)|[link](https://github.com/ZhAnGToNG1/transfer_learning_cspt)|
|**RVSA**|**Advancing plain vision transformer toward remote sensing foundation model**|TGRS2022|[RVSA](https://ieeexplore.ieee.org/abstract/document/9956816)|[link](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA)|
|**SatMAE**|**SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery**|NeurIPS2022|[SatMAE](https://proceedings.neurips.cc/paper_files/paper/2022/hash/01c561df365429f33fcd7a7faa44c985-Abstract-Conference.html)|[link](https://github.com/sustainlab-group/SatMAE)|
|**DINO-MC**|**Extending Global-Local View Alignment for Self-Supervised Learning with Remote Sensing Imagery**|Arxiv2023|[DINO-MC](https://arxiv.org/abs/2303.06670)|[link](https://github.com/WennyXY/DINO-MC)|
|**CMID**|**CMID: A Unified Self-Supervised Learning Framework for Remote Sensing Image Understanding**|TGRS2023|[CMID](https://ieeexplore.ieee.org/abstract/document/10105625)|[link](https://github.com/NJU-LHRS/official-CMID)|
|**Presto**|**Lightweight, Pre-trained Transformers for Remote Sensing Timeseries**|Arxiv2023|[Presto](https://arxiv.org/abs/2304.14065)|[link](https://github.com/nasaharvest/presto)|
|**AST**|**AST: Adaptive Self-supervised Transformer for Optical Remote Sensing Representation**|ISPRS JPRS2023|[AST](https://doi.org/10.1016/j.isprsjprs.2023.04.003)|null|
|**TOV**|**TOV: The original vision model for optical remote sensing image understanding via self-supervised learning**|JSTARS2023|[TOV](https://ieeexplore.ieee.org/abstract/document/10110958)|[link](https://github.com/GeoX-Lab/G-RSIM/tree/main/TOV_v1)|
|**CACo**|**Change-Aware Sampling and Contrastive Learning for Satellite Images**|CVPR2023|[CACo](https://openaccess.thecvf.com/content/CVPR2023/html/Mall_Change-Aware_Sampling_and_Contrastive_Learning_for_Satellite_Images_CVPR_2023_paper.html)|[link](https://github.com/utkarshmall13/CACo)|
|**IaI-SimCLR**|**Multi-Modal Multi-Objective Contrastive Learning for Sentinel-1/2 Imagery**|CVPRW2023|[IaI-SimCLR](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/html/Prexl_Multi-Modal_Multi-Objective_Contrastive_Learning_for_Sentinel-12_Imagery_CVPRW_2023_paper.html)|null|
|**-**|**A Self-Supervised Cross-Modal Remote Sensing Foundation Model with Multi-Domain Representation and Cross-Domain Fusion**|IGARSS2023|[Paper](https://ieeexplore.ieee.org/abstract/document/10282433)|null|
|**SatLas**|**SatlasPretrain: A Large-Scale Dataset for Remote Sensing Image Understanding**|ICCV2023|[SatLas](https://doi.org/10.1109/iccv51070.2023.01538)|[link](https://github.com/allenai/satlas)|
|**GFM**|**Towards Geospatial Foundation Models via Continual Pretraining**|ICCV2023|[GFM](https://doi.org/10.1109/iccv51070.2023.01541)|[link](https://github.com/mmendiet/GFM)|
|**Scale-MAE**|**Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning**|ICCV2023|[Scale-MAE](https://doi.org/10.1109/iccv51070.2023.00378)|[link](https://github.com/bair-climate-initiative/scale-mae)|
|**Prithvi**|**Foundation Models for Generalist Geospatial Artificial Intelligence**|Arxiv2023|[Prithvi](https://arxiv.org/abs/2310.18660)|[link](https://huggingface.co/ibm-nasa-geospatial)|
|**RingMo-Sense**|**RingMo-Sense: Remote Sensing Foundation Model for Spatiotemporal Prediction via Spatiotemporal Evolution Disentangling**|TGRS2023|[RingMo-Sense](https://ieeexplore.ieee.org/abstract/document/10254320)|null|
|**EarthPT**|**EarthPT: a time series foundation model for Earth Observation**|NeurIPS2023 CCAI workshop|[EarthPT](https://arxiv.org/abs/2309.07207)|[link](https://github.com/aspiaspace/EarthPT)|
|**CROMA**|**CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders**|NeurIPS2023|[CROMA](https://arxiv.org/abs/2311.00566)|[link](https://github.com/antofuller/CROMA)|
|**Cross-Scale MAE**|**Cross-Scale MAE: A Tale of Multiscale Exploitation in Remote Sensing**|NeurIPS2023|[Cross-Scale MAE](https://openreview.net/pdf?id=5oEVdOd6TV)|[link](https://github.com/aicip/Cross-Scale-MAE)|
|**USat**|**USat: A Unified Self-Supervised Encoder for Multi-Sensor Satellite Imagery**|Arxiv2023|[USat](https://arxiv.org/abs/2312.02199)|[link](https://github.com/stanfordmlgroup/USat)|
|**AIEarth**|**Analytical Insight of Earth: A Cloud-Platform of Intelligent Computing for Geospatial Big Data**|Arxiv2023|[AIEarth](https://arxiv.org/abs/2312.16385)|[link](https://engine-aiearth.aliyun.com/#/)|
|**GeRSP**|**Generic Knowledge Boosted Pretraining for Remote Sensing Images**|TGRS2024|[GeRSP](https://doi.org/10.1109/tgrs.2024.3354031)|[GeRSP](https://github.com/floatingstarZ/GeRSP)|
|**SMLFR**|**Generative ConvNet Foundation Model With Sparse Modeling and Low-Frequency Reconstruction for Remote Sensing Image Interpretation**|TGRS2024|[SMLFR](https://ieeexplore.ieee.org/abstract/document/10378718)|[link](https://github.com/HIT-SIRS/SMLFR)|
|**RingMo-lite**|**RingMo-Lite: A Remote Sensing Lightweight Network With CNN-Transformer Hybrid Framework**|IEEE TGRS2024|[RingMo-lite](https://doi.org/10.1109/tgrs.2024.3360447)|null|
|**U-BARN**|**Self-Supervised Spatio-Temporal Representation Learning of Satellite Image Time Series**|JSTARS2024|[Paper](https://ieeexplore.ieee.org/document/10414422)|[link](https://src.koda.cnrs.fr/iris.dumeur/ssl_ubarn)|
|**SpectralGPT**|**SpectralGPT: Spectral Remote Sensing Foundation Model**|TPAMI2024|[SpectralGPT](https://doi.org/10.1109/tpami.2024.3362475)|[link](https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT)|
|**SwiMDiff**|**SwiMDiff: Scene-Wide Matching Contrastive Learning With Diffusion Constraint for Remote Sensing Image**|TGRS2024|[SwiMDiff](https://doi.org/10.1109/tgrs.2024.3371481)|null|
|**DOFA**|**Neural Plasticity-Inspired Multimodal Foundation Model for Earth Observation**|Arxiv2024|[DOFA](https://arxiv.org/abs/2403.15356)|[link](https://github.com/zhu-xlab/DOFA)|
|**-**|**Masked Feature Modeling for Generative Self-Supervised Representation Learning of High-Resolution Remote Sensing Images**|IEEE JSTARS2024|[Paper](https://doi.org/10.1109/jstars.2024.3385420)|null|
|**BFM**|**A Billion-scale Foundation Model for Remote Sensing Images**|IEEE JSTARS2024|[BFM](https://doi.org/10.1109/jstars.2024.3401772)|null|
|**Clay**|**Clay Foundation Model**|Arxiv2024|null|[link](https://clay-foundation.github.io/model/)|
|**Hydro**|**Hydro--A Foundation Model for Water in Satellite Imagery**|Arxiv2024|null|[link](https://github.com/isaaccorley/hydro-foundation-model)|
|**S2MAE**|**S2MAE: A Spatial-Spectral Pretraining Foundation Model for Spectral Remote Sensing Data**|CVPR2024|[S2MAE](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_S2MAE_A_Spatial-Spectral_Pretraining_Foundation_Model_for_Spectral_Remote_Sensing_CVPR_2024_paper.pdf)|null|
|**SatMAE++**|**Rethinking Transformers Pre-training for Multi-Spectral Satellite Imagery**|CVPR2024|[SatMAE++](https://doi.org/10.1109/cvpr52733.2024.02627)|[link](https://github.com/techmn/satmae_pp)|
|**msGFM**|**Bridging Remote Sensors with Multisensor Geospatial Foundation Models**|CVPR2024|[msGFM](https://doi.org/10.1109/cvpr52733.2024.02631)|[link](https://github.com/boranhan/Geospatial_Foundation_Models)|
|**SkySense**|**SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery**|CVPR2024|[SkySense](https://openaccess.thecvf.com/content/CVPR2024/html/Guo_SkySense_A_Multi-Modal_Remote_Sensing_Foundation_Model_Towards_Universal_Interpretation_CVPR_2024_paper.html)|[link](https://github.com/Jack-bo1220/SkySense)|
|**MTP**|**MTP: Advancing Remote Sensing Foundation Model via Multi-Task Pretraining**|IEEE JSTARS2024|[MTP](https://doi.org/10.1109/jstars.2024.3408154)|[link](https://github.com/ViTAE-Transformer/MTP)|
|**RS-DFM**|**RS-DFM: A Remote Sensing Distributed Foundation Model for Diverse Downstream Tasks**|Arxiv2024|[RS-DFM](https://arxiv.org/abs/2406.07032)|null|
|**OFA-Net**|**One for All: Toward Unified Foundation Models for Earth Vision**|IGARSS2024|[OFA-Net](https://doi.org/10.1109/igarss53475.2024.10641637)|null|
|**-**|**Lightweight and Efficient: A Family of Multimodal Earth Observation Foundation Models**|IGARSS2024|[Paper](https://doi.org/10.1109/igarss53475.2024.10641132)|null|
|**MM-VSF**|**Towards Knowledge Guided Pretraining Approaches for Multimodal Foundation Models: Applications in Remote Sensing**|Arxiv2024|[MM-VSF](https://arxiv.org/abs/2407.19660)|null|
|**LeMeViT**|**LeMeViT: Efficient Vision Transformer with Learnable Meta Tokens for Remote Sensing Image Interpretation**|IJCAI2024|[LeMeViT](https://arxiv.org/abs/2405.09789)|[link](https://github.com/ViTAE-Transformer/LeMeViT/tree/main?tab=readme-ov-file)|
|**SAR-JEPA**|**Predicting Gradient is Better: Exploring Self-Supervised Learning for SAR ATR with a Joint-Embedding Predictive Architecture**|ISPRS JPRS2024|[SAR-JEPA](https://www.sciencedirect.com/science/article/pii/S0924271624003514)|[link](https://github.com/waterdisappear/SAR-JEPA)|
|**DeCUR**|**DeCUR: decoupling common & unique representations for multimodal self-supervision**|ECCV2024|[DeCUR](https://doi.org/10.1007/978-3-031-73397-0_17)|[link](https://github.com/zhu-xlab/DeCUR)|
|**MMEarth**|**MMEarth: Exploring Multi-Modal Pretext Tasks For Geospatial Representation Learning**|ECCV2024|[MMEarth](https://doi.org/10.1007/978-3-031-73039-9_10)|[link](https://vishalned.github.io/mmearth/)|
|**OmniSat**|**OmniSat: Self-Supervised Modality Fusion for Earth Observation**|ECCV2024|[OmniSat](https://doi.org/10.1007/978-3-031-73390-1_24)|[link](https://github.com/gastruc/OmniSat?tab=readme-ov-file)|
|**MA3E**|**Masked Angle-Aware Autoencoder for Remote Sensing Images**|ECCV2024|[MA3E](https://doi.org/10.1007/978-3-031-73242-3_15)|[link](https://github.com/benesakitam/MA3E)|
|**SoftCon**|**Multi-Label Guided Soft Contrastive Learning for Efficient Earth Observation Pretraining**|TGRS2024|[SoftCon](https://ieeexplore.ieee.org/abstract/document/10726860)|[link](https://github.com/zhu-xlab/softcon?tab=readme-ov-file)|
|**PIS**|**Pretrain a Remote Sensing Foundation Model by Promoting Intra-instance Similarity**|TGRS2024|[PIS](https://ieeexplore.ieee.org/abstract/document/10697182)|[link](https://github.com/ShawnAn-WHU/PIS)|
|**FG-MAE**|**Feature Guided Masked Autoencoder for Self-Supervised Learning in Remote Sensing**|IEEE JSTARS2024|[FG-MAE](https://doi.org/10.1109/jstars.2024.3493237)|[link](https://github.com/zhu-xlab/FGMAE)|
|**-**|**A Multimodal Unified Representation Learning Framework With Masked Image Modeling for Remote Sensing Images**|IEEE TGRS2024|[Paper](https://doi.org/10.1109/tgrs.2024.3494244)|null|
|**OReole-FM**|**OReole-FM: successes and challenges toward billion-parameter foundation models for high-resolution satellite imagery**|SIGSPATIAL2024|[OReole-FM](https://doi.org/10.1145/3678717.3691292)|null|
|**SatVision-TOA**|**SatVision-TOA: A Geospatial Foundation Model for Coarse-Resolution All-Sky Remote Sensing Imagery**|Arxiv2024|[SatVision-TOA](https://arxiv.org/abs/2411.17000)|[link](https://github.com/nasa-nccs-hpda/pytorch-caney)|
|**Prithvi-EO-2.0**|**Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications**|Arxiv2024|[Prithvi-EO-2.0](https://arxiv.org/abs/2412.02732)|[link](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)|
|**A2-MAE**|**A2-MAE: A spatial-temporal-spectral unified remote sensing pre-training method based on anchor-aware masked autoencoder**|IEEE TGRS2025|[A2-MAE](https://doi.org/10.1109/tgrs.2025.3571123)|null|
|**FoMo**|**FoMo: Multi-Modal, Multi-Scale and Multi-Task Remote Sensing Foundation Models for Forest Monitoring**|AAAI2025|[FoMo](https://doi.org/10.1609/aaai.v39i27.35002)|[link](https://github.com/RolnickLab/FoMo-Bench)|
|**PIEViT**|**Pattern Integration and Enhancement Vision Transformer for Self-Supervised Learning in Remote Sensing**|IEEE TGRS2025|[PIEViT](https://doi.org/10.1109/tgrs.2025.3541390)|null|
|**SARATR-X**|**SARATR-X: Toward Building a Foundation Model for SAR Target Recognition**|IEEE TIP2025|[SARATR-X](https://ieeexplore.ieee.org/abstract/document/10856784)|[link](https://github.com/waterdisappear/SARATR-X)|
|**SatMamba**|**SatMamba: Development of Foundation Models for Remote Sensing Imagery Using State Space Models**|Arxiv2025|[SatMamba](https://arxiv.org/abs/2502.00435)|[link](https://github.com/mdchuc/HRSFM)|
|**DynamicVis**|**DynamicVis: Dynamic Visual Perception for Efficient Remote Sensing Foundation Models**|Arxiv2025|[DynamicVis](https://arxiv.org/abs/2503.16426)|[link](https://github.com/KyanChen/DynamicVis)|
|**SUMMIT**|**SUMMIT: A SAR foundation model with multiple auxiliary tasks enhanced intrinsic characteristics**|IJAEO2025|[SUMMIT](https://doi.org/10.1016/j.jag.2025.104624)|null|
|**SenPa-MAE**|**SenPa-MAE: Sensor Parameter Aware Masked Autoencoder for Multi-Satellite Self-Supervised Pretraining**|LNCS2025|[SenPa-MAE](https://doi.org/10.1007/978-3-031-85187-2_20)|[link](https://github.com/JonathanPrexl/SenPa-MAE)|
|**HyperSIGMA**|**HyperSIGMA: Hyperspectral Intelligence Comprehension Foundation Model**|IEEE TPAMI2025|[HyperSIGMA](https://arxiv.org/abs/2406.11519)|[link](https://github.com/WHU-Sigma/HyperSIGMA?tab=readme-ov-file)|
|**HyperSL**|**HyperSL: A Spectral Foundation Model for Hyperspectral Image Interpretation**|IEEE TGRS2025|[HyperSL](https://ieeexplore.ieee.org/abstract/document/10981753)|[link](https://github.com/kkweil/HyperSL)|
|**TiMo**|**TiMo: Spatiotemporal Foundation Model for Satellite Image Time Series**|Arxiv2025|[TiMo](https://arxiv.org/abs/2505.08723)|[link](https://github.com/MiliLab/TiMo)|
|**Panopticon**|**Panopticon: Advancing Any-Sensor Foundation Models for Earth Observation**|CVPRW2025 (EarthVision Best Paper)|[Panopticon](https://arxiv.org/abs/2503.10845)|[link](https://github.com/Panopticon-FM/panopticon)|
|**HyperFree**|**HyperFree: A Channel-adaptive and Tuning-free Foundation Model for Hyperspectral Remote Sensing Imagery**|CVPR2025|[HyperFree](https://rsidea.whu.edu.cn/HyperFree.pdf)|[link](https://github.com/Jingtao-Li-CVer/HyperFree)|
|**SpectralEarth**|**SpectralEarth: Training Hyperspectral Foundation Models at Scale**|IEEE JSTARS2025|[SpectralEarth](https://doi.org/10.1109/jstars.2025.3581451)|[link](https://github.com/zhu-xlab/softcon)|
|**WV-Net**|**WV-Net: A foundation model for SAR WV-mode satellite imagery trained using contrastive self-supervised learning on 10 million images**|AIES2025|[WV-Net](https://arxiv.org/abs/2406.18765)|null|
|**AnySat**|**AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities**|CVPR2025|[AnySat](https://arxiv.org/abs/2412.14123)|[link](https://github.com/gastruc/AnySat)|
|**TerraFM**|**TerraFM: A Scalable Foundation Model for Unified Multisensor Earth Observation**|Arxiv2025|[TerraFM](https://arxiv.org/abs/2506.06281)|[link](https://github.com/mbzuai-oryx/TerraFM)|
|**Galileo**|**Galileo: Learning Global & Local Features of Many Remote Sensing Modalities**|ICML2025 TerraBytes Workshop|[Galileo](https://arxiv.org/abs/2502.09356)|[link](https://github.com/nasaharvest/galileo)|
|**DeepAndes**|**DeepAndes: A Self-Supervised Vision Foundation Model for Multispectral Remote Sensing Imagery of the Andes**|IEEE JSTARS2025|[DeepAndes](https://doi.org/10.1109/jstars.2025.3619423)|[link](https://github.com/geopacha/DeepAndes)|
|**RingMamba**|**RingMamba: Remote Sensing Multisensor Pretraining With Visual State Space Model**|IEEE TGRS2025|[RingMamba](https://doi.org/10.1109/tgrs.2025.3603998)|[link](https://github.com/ningerhhh/RIFR)|
|**RingMo-Aerial**|**RingMo-Aerial: An Aerial Remote Sensing Foundation Model With Affine Transformation Contrastive Learning**|IEEE TPAMI2025|[RingMo-Aerial](https://doi.org/10.1109/tpami.2025.3602237)|null|
|**SeaMo**|**SeaMo: A Multi-Seasonal and Multimodal Remote Sensing Foundation Model**|Information Fusion2025|[SeaMo](https://www.sciencedirect.com/science/article/pii/S1566253525004075)|null|
|**MoSAiC**|**MoSAiC: Multi-Modal Multi-Label Supervision-Aware Contrastive Learning for Remote Sensing**|IEEE Sensors Journal 2025|[MoSAiC](https://arxiv.org/abs/2507.08683)|null|
|**CGEarthEye**|**CGEarthEye: A High-Resolution Remote Sensing Vision Foundation Model Based on the Jilin-1 Satellite Constellation**|Arxiv2025|[CGEarthEye](https://arxiv.org/abs/2507.00356)|null|
|**AlphaEarth**|**AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data**|Arxiv2025|[AlphaEarth](https://arxiv.org/abs/2507.22291)|[link](https://deepmind.google/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/)|
|**SkySense++**|**A semantic-enhanced multi-modal remote sensing foundation model for Earth observation**|Nature Machine Intelligence 2025|[SkySense++](https://www.nature.com/articles/s42256-025-01078-8)|[link](https://github.com/kang-wu/SkySensePlusPlus?tab=readme-ov-file)|
|**CtxMIM**|**CtxMIM: Context-Enhanced Masked Image Modeling for Remote Sensing Image Understanding**|ACM TOMM2025|[CtxMIM](https://doi.org/10.1145/3769084)|null|
|**ViTP**|**Visual Instruction Pretraining for Domain-Specific Foundation Models**|Arxiv2025|[ViTP](https://arxiv.org/abs/2509.17562)|[link](https://huggingface.co/GreatBird/ViTP)|
|**SatDiFuser**|**Can Generative Geospatial Diffusion Models Excel as Discriminative Geospatial Foundation Models?**|ICCV2025|[SatDiFuser](https://arxiv.org/abs/2503.07890)|[link](https://github.com/yurujaja/SatDiFuser)|
|**FedSense**|**Towards Privacy-preserved Pre-training of Remote Sensing Foundation Models with Federated Mutual-guidance Learning**|ICCV2025|[FedSense](https://arxiv.org/abs/2503.11051)|null|
|**Copernicus-FM**|**Towards a Unified Copernicus Foundation Model for Earth Vision**|ICCV2025|[Copernicus-FM](https://arxiv.org/abs/2503.11849)|[link](https://github.com/zhu-xlab/Copernicus-FM)|
|**TerraMind**|**TerraMind: Large-Scale Generative Multimodality for Earth Observation**|ICCV2025|[TerraMind](https://arxiv.org/abs/2504.11171)|[link](https://github.com/IBM/terramind)|
|**SelectiveMAE**|**Harnessing Massive Satellite Imagery with Efficient Masked Image Modeling**|ICCV2025|[SelectiveMAE](https://arxiv.org/abs/2406.11933)|[link](https://github.com/Fengxiang23/SelectiveMAE)|
|**SMARTIES**|**SMARTIES: Spectrum-Aware Multi-Sensor Auto-Encoder for Remote Sensing Images**|ICCV2025|[SMARTIES](https://arxiv.org/abs/2506.19585)|[link](https://gsumbul.github.io/SMARTIES/)|
|**SkySense V2**|**SkySense V2: A Unified Foundation Model for Multi-modal Remote Sensing**|ICCV2025|[SkySense V2](https://arxiv.org/abs/2507.13812)|null|
|**RS-vHeat**|**RS-vHeat: Heat Conduction Guided Efficient Remote Sensing Foundation Model**|ICCV2025|[RS-vHeat](https://arxiv.org/abs/2411.17984)|null|
|**RoMA**|**RoMA: Scaling up Mamba-based Foundation Models for Remote Sensing**|NeurIPS2025|[RoMA](https://arxiv.org/abs/2503.10392)|[link](https://github.com/MiliLab/RoMA)|
|**GeoLink**|**GeoLink: Empowering Remote Sensing Foundation Model with OpenStreetMap Data**|NeurIPS2025|[GeoLink](https://arxiv.org/abs/2509.26016)|[link](https://github.com/bailubin/GeoLink_NeurIPS2025)|
|**CrossEarth**|**CrossEarth: Geospatial Vision Foundation Model for Domain Generalizable Remote Sensing Semantic Segmentation**|IEEE TPAMI2025|[CrossEarth](https://doi.org/10.1109/tpami.2025.3649001)|[link](https://github.com/Cuzyoung/CrossEarth)|
|**PhySwin**|**PhySwin: An Efficient and Physically-Informed Foundation Model for Multispectral Earth Observation**|NeurIPS2025|[PhySwin](https://openreview.net/forum?id=zrBucj9BwG)|null|
|**FlexiMo**|**FlexiMo: A Flexible Remote Sensing Foundation Model**|IEEE TGRS2026|[FlexiMo](https://doi.org/10.1109/tgrs.2026.3656362)|null|
|**MAPEX**|**MAPEX: Modality-Aware Pruning of Experts for Remote Sensing Foundation Models**|IEEE TGRS2026|[MAPEX](https://doi.org/10.1109/tgrs.2026.3652100)|[link](https://github.com/HSG-AIML/MAPEX)|
|**-**|**A Complex-Valued SAR Foundation Model Based on Physically Inspired Representation Learning**|IEEE TIP2026|[Paper](https://doi.org/10.1109/tip.2026.3652417)|null|
|**MAESTRO**|**MAESTRO: Masked AutoEncoders for Multimodal, Multitemporal, and Multispectral Earth Observation Data**|WACV2026|[MAESTRO](https://openaccess.thecvf.com/content/WACV2026/papers/Labatie_MAESTRO_Masked_AutoEncoders_for_Multimodal_Multitemporal_and_Multispectral_Earth_Observation_WACV_2026_paper.pdf)|[link](https://github.com/IGNF/MAESTRO)|
|**RingMoE**|**RingMoE: Mixture-of-Modality-Experts Multi-Modal Foundation Models for Universal Remote Sensing Image Interpretation**|IEEE TPAMI2026|[RingMoE](https://doi.org/10.1109/tpami.2025.3643453)|null|
|**Alliance**|**Alliance: All-in-One Spectral-Spatial-Frequency Awareness Foundation Model**|IEEE TPAMI2026|[Alliance](https://doi.org/10.1109/tpami.2025.3639595)|null|
|**THOR**|**THOR: A Versatile Foundation Model for Earth Observation Climate and Society Applications**|Arxiv2026|[THOR](https://arxiv.org/abs/2601.16011)|[link](https://github.com/FM4CS/THOR)|
|**AgriFM**|**AgriFM: A multi-source temporal remote sensing foundation model for Agriculture mapping**|RSE2026|[Paper](https://doi.org/10.1016/j.rse.2026.115234)|[link](https://github.com/flyakon/AgriFM)|
|**SIGMAE**|**SIGMAE: A Spectral-Index-Guided Foundation Model for Multispectral Remote Sensing**|Arxiv2026|[SIGMAE](https://arxiv.org/abs/2603.07463)|[link](https://github.com/zxk688/SIGMAE)|
|**CrossEarth-SAR**|**CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation**|Arxiv2026|[CrossEarth-SAR](https://arxiv.org/abs/2603.12008)|[link](https://github.com/VisionXLab/CrossEarth-SAR)|
|**NeighborMAE**|**NeighborMAE: Exploiting Spatial Dependencies between Neighboring Earth Observation Images in Masked Autoencoders Pretraining**|CVPR2026|[NeighborMAE](https://arxiv.org/abs/2603.02522)|null|
|**MOMO**|**MOMO: Mars Orbital Model Foundation Model for Mars Orbital Applications**|CVPR2026|[MOMO](https://arxiv.org/abs/2604.02719)|[link](https://github.com/kerner-lab/MOMO)|
|**TESSERA**|**TESSERA: Temporal Embeddings of Surface Spectra for Earth Representation and Analysis**|CVPR2026|[TESSERA](https://arxiv.org/abs/2506.20380)|[link](https://github.com/ucam-eo/tessera)|
|**OlmoEarth**|**OlmoEarth: Stable Latent Image Modeling for Multimodal Earth Observation**|CVPR2026|[OlmoEarth](https://arxiv.org/abs/2511.13655)|[link](https://github.com/allenai/olmoearth_pretrain)|
|**RAMEN**|**RAMEN: Resolution-Adjustable Multimodal Encoder for Earth Observation**|CVPR2026|[RAMEN](https://arxiv.org/abs/2512.05025)|[link](https://github.com/nicolashoudre/RAMEN)|
|**SARMAE**|**SARMAE: Masked Autoencoder for SAR Representation Learning**|CVPR2026|[SARMAE](https://arxiv.org/abs/2512.16635)|[link](https://github.com/MiliLab/SARMAE)|
## Remote Sensing <ins>Vision-Language</ins> Foundation Models

|Abbreviation|Title|Publication|Paper|Code & Weights|
|:---:|---|:---:|:---:|:---:|
|**-**|**Charting New Territories: Exploring the Geographic and Geospatial Capabilities of Multimodal LLMs**|Arxiv2023|[Paper](https://arxiv.org/abs/2311.14656)|[link](https://github.com/jonathan-roberts1/charting-new-territories)|
|**-**|**Remote Sensing ChatGPT: Solving Remote Sensing Tasks with ChatGPT and Visual Models**|Arxiv2024|[Paper](https://arxiv.org/abs/2401.09083)|[link](https://github.com/HaonanGuo/Remote-Sensing-ChatGPT)|
|**SkyCLIP**|**SkyScript: A Large and Semantically Diverse Vision-Language Dataset for Remote Sensing**|AAAI2024|[SkyCLIP](https://arxiv.org/abs/2312.12856)|[link](https://github.com/wangzhecheng/SkyScript)|
|**GeoRSCLIP**|**RS5M and GeoRSCLIP: A Large Scale Vision-Language Dataset and A Large Vision-Language Model for Remote Sensing**|IEEE TGRS2024|[GeoRSCLIP](https://arxiv.org/abs/2306.11300)|[link](https://github.com/om-ai-lab/RS5M)|
|**RemoteCLIP**|**RemoteCLIP: A Vision Language Foundation Model for Remote Sensing**|IEEE TGRS2024|[RemoteCLIP](https://arxiv.org/abs/2306.11029)|[link](https://github.com/ChenDelong1999/RemoteCLIP)|
|**EarthGPT**|**EarthGPT: A Universal Multimodal Large Language Model for Multisensor Image Comprehension in Remote Sensing Domain**|IEEE TGRS2024|[EarthGPT](https://doi.org/10.1109/tgrs.2024.3409624)|[link](https://github.com/wivizhang/EarthGPT)|
|**GRAFT**|**Remote Sensing Vision-Language Foundation Models without Annotations via Ground Remote Alignment**|ICLR2024|[GRAFT](https://openreview.net/pdf?id=w9tc699w3Z)|null|
|**EarthMarker**|**EarthMarker: A Visual Prompting Multi-modal Large Language Model for Remote Sensing**|IEEE TGRS2024|[EarthMarker](https://arxiv.org/abs/2407.13596)|[link](https://github.com/wivizhang/EarthMarker)|
|**RingMoGPT**|**RingMoGPT: A Unified Remote Sensing Foundation Model for Vision, Language, and Grounded Tasks**|IEEE TGRS2024|[RingMoGPT](https://ieeexplore.ieee.org/document/10777289)|null|
|**RS-LLaVA**|**RS-LLaVA: Large Vision Language Model for Joint Captioning and Question Answering in Remote Sensing Imagery**|RS2024|[RS-LLaVA](https://www.mdpi.com/2072-4292/16/9/1477)|[link](https://github.com/BigData-KSU/RS-LLaVA)|
|**GeoChat**|**GeoChat: Grounded Large Vision-Language Model for Remote Sensing**|CVPR2024|[GeoChat](https://openaccess.thecvf.com/content/CVPR2024/html/Kuckreja_GeoChat_Grounded_Large_Vision-Language_Model_for_Remote_Sensing_CVPR_2024_paper.html)|[link](https://github.com/mbzuai-oryx/GeoChat)|
|**SkySenseGPT**|**SkySenseGPT: A Fine-Grained Instruction Tuning Dataset and Model for Remote Sensing Vision-Language Understanding**|Arxiv2024|[SkySenseGPT](https://arxiv.org/abs/2406.10100)|[link](https://github.com/Luo-Z13/SkySenseGPT)|
|**RSCLIP**|**Pushing the Limits of Vision-Language Models in Remote Sensing without Human Annotations**|Arxiv2024|[RSCLIP](https://arxiv.org/abs/2409.07048)|null|
|**GeoText**|**Towards Natural Language-Guided Drones: GeoText-1652 Benchmark with Spatial Relation Matching**|ECCV2024|[GeoText](https://arxiv.org/abs/2311.12751)|[link](https://github.com/MultimodalGeo/GeoText-1652)|
|**LHRS-Bot**|**LHRS-Bot: Empowering Remote Sensing with VGI-Enhanced Large Multimodal Language Model**|ECCV2024|[LHRS-Bot](https://arxiv.org/abs/2402.02544)|[link](https://github.com/NJU-LHRS/LHRS-Bot)|
|**GeoGround**|**GeoGround: A Unified Large Vision-Language Model for Remote Sensing Visual Grounding**|Arxiv2024|[GeoGround](https://arxiv.org/abs/2411.11904)|[link](https://github.com/zytx121/GeoGround)|
|**RSUniVLM**|**RSUniVLM: A Unified Vision Language Model for Remote Sensing via Granularity-oriented Mixture of Experts**|Arxiv2024|[RSUniVLM](https://arxiv.org/abs/2412.05679)|null|
|**REO-VLM**|**REO-VLM: Transforming VLM to Meet Regression Challenges in Earth Observation**|Arxiv2024|[REO-VLM](https://arxiv.org/abs/2412.16583)|null|
|**UniRS**|**UniRS: Unifying Multi-temporal Remote Sensing Tasks through Vision Language Models**|Arxiv2024|[UniRS](https://arxiv.org/abs/2412.20742)|null|
|**VHM**|**VHM: Versatile and Honest Vision Language Model for Remote Sensing Image Analysis**|AAAI2025|[VHM](https://ojs.aaai.org/index.php/AAAI/article/view/32683)|[link](https://github.com/opendatalab/VHM)|
|**-**|**Quality-Driven Curation of Remote Sensing Vision-Language Data via Learned Scoring Models**|Arxiv2025|[Paper](https://arxiv.org/abs/2503.00743)|null|
|**DOFA-CLIP**|**DOFA-CLIP: Multimodal Vision-Language Foundation Models for Earth Observation**|Arxiv2025|[DOFA-CLIP](https://arxiv.org/abs/2503.06312)|[link](https://github.com/xiong-zhitong/GeoLB-SigLIP)|
|**Falcon**|**Falcon: A Remote Sensing Vision-Language Foundation Model**|Arxiv2025|[Falcon](https://arxiv.org/abs/2503.11070)|[link](https://github.com/TianHuiLab/Falcon)|
|**GeoRSMLLM**|**GeoRSMLLM: A Multimodal Large Language Model for Vision-Language Tasks in Geoscience and Remote Sensing**|Arxiv2025|[GeoRSMLLM](https://arxiv.org/abs/2503.12490)|null|
|**OmniGeo**|**OmniGeo: Towards a Multimodal Large Language Models for Geospatial Artificial Intelligence**|Arxiv2025|[OmniGeo](https://arxiv.org/abs/2503.16326)|null|
|**DGTRS-CLIP**|**DGTRSD & DGTRS-CLIP: A Dual-Granularity Remote Sensing Image-Text Dataset and Vision Language Foundation Model for Alignment**|Arxiv2025|[DGTRS-CLIP](https://arxiv.org/abs/2503.19311)|[link](https://github.com/MitsuiChen14/DGTRS)|
|**EagleVision**|**EagleVision: Object-level Attribute Multimodal LLM for Remote Sensing**|Arxiv2025|[EagleVision](https://arxiv.org/abs/2503.23330)|[link](https://github.com/XiangTodayEatsWhat/EagleVision)|
|**SkyEyeGPT**|**SkyEyeGPT: Unifying Remote Sensing Vision-Language Tasks via Instruction Tuning with Large Language Model**|ISPRS JPRS2025|[SkyEyeGPT](https://doi.org/10.1016/j.isprsjprs.2025.01.020)|[link](https://github.com/ZhanYang-nwpu/SkyEyeGPT)|
|**SegEarth-R1**|**SegEarth-R1: Geospatial Pixel Reasoning via Large Language Model**|Arxiv2025|[SegEarth-R1](https://arxiv.org/abs/2504.09644)|[link](https://github.com/earth-insights/SegEarth-R1)|
|**EarthGPT-X**|**EarthGPT-X: A Spatial MLLM for Multi-level Multi-Source Remote Sensing Imagery Understanding with Visual Prompting**|Arxiv2025|[EarthGPT-X](https://arxiv.org/abs/2504.12795)|[link](https://github.com/wivizhang/EarthGPT-X)|
|**TEOChat**|**TEOChat: A Large Vision-Language Assistant for Temporal Earth Observation Data**|ICLR2025|[TEOChat](https://arxiv.org/abs/2410.06234)|[link](https://github.com/ermongroup/TEOChat)|
|**LISAt**|**LISAt: Language-Instructed Segmentation Assistant for Satellite Imagery**|Arxiv2025|[LISAt](https://arxiv.org/abs/2505.02829)|null|
|**DynamicVL**|**DynamicVL: Benchmarking Multimodal Large Language Models for Dynamic City Understanding**|Arxiv2025|[DynamicVL](https://arxiv.org/abs/2505.21076)|null|
|**RSGPT**|**RSGPT: A Remote Sensing Vision Language Model and Benchmark**|ISPRS JPRS2025|[RSGPT](https://doi.org/10.1016/j.isprsjprs.2025.04.001)|[link](https://github.com/Lavender105/RSGPT)|
|**LHRS-Bot-Nova**|**LHRS-Bot-Nova: Improved Multimodal Large Language Model for Remote Sensing Vision-Language Interpretation**|ISPRS JPRS2025|[LHRS-Bot-Nova](https://doi.org/10.1016/j.isprsjprs.2025.06.003)|[link](https://github.com/NJU-LHRS/LHRS-Bot)|
|**EarthDial**|**EarthDial: Turning Multi-sensory Earth Observations to Interactive Dialogues**|CVPR2025|[EarthDial](https://openaccess.thecvf.com/content/CVPR2025/html/Soni_EarthDial_Turning_Multi-sensory_Earth_Observations_to_Interactive_Dialogues_CVPR_2025_paper.html)|[link](https://github.com/hiyamdebary/EarthDial)|
|**SkySense-O**|**SkySense-O: Towards Open-World Remote Sensing Interpretation with Vision-Centric Visual-Language Modeling**|CVPR2025|[SkySense-O](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_SkySense-O_Towards_Open-World_Remote_Sensing_Interpretation_with_Vision-Centric_Visual-Language_Modeling_CVPR_2025_paper.pdf)|[link](https://github.com/zqcrafts/SkySense-O)|
|**XLRS-Bench**|**XLRS-Bench: Could Your Multimodal LLMs Understand Extremely Large Ultra-High-Resolution Remote Sensing Imagery?**|CVPR2025|[XLRS-Bench](https://arxiv.org/abs/2503.23771)|null|
|**GeoPix**|**GeoPix: Multi-Modal Large Language Model for Pixel-level Image Understanding in Remote Sensing**|IEEE GRSM2025|[GeoPix](https://arxiv.org/abs/2501.06828)|[link](https://github.com/Norman-Ou/GeoPix)|
|**Co-LLaVA**|**Co-LLaVA: Efficient Remote Sensing Visual Question Answering via Model Collaboration**|RS2025|[Co-LLaVA](https://doi.org/10.3390/rs17030466)|null|
|**RLita**|**RLita: A Region-Level Image-Text Alignment Method for Remote Sensing Foundation Model**|RS2025|[RLita](https://doi.org/10.3390/rs17101661)|null|
|**EarthMind**|**EarthMind: Leveraging Cross-Sensor Data for Advanced Earth Observation Interpretation with a Unified Multimodal LLM**|Arxiv2025|[EarthMind](https://arxiv.org/abs/2506.01667)|[link](https://github.com/shuyansy/EarthMind)|
|**-**|**Remote Sensing Large Vision-Language Model: Semantic-augmented Multi-level Alignment and Semantic-aware Expert Modeling**|Arxiv2025|[Paper](https://arxiv.org/abs/2506.21863)|null|
|**GeoPixel**|**GeoPixel: Pixel Grounding Large Multimodal Model in Remote Sensing**|ICML2025|[GeoPixel](https://arxiv.org/abs/2501.13925)|[link](https://github.com/mbzuai-oryx/GeoPixel)|
|**RingMo-Agent**|**RingMo-Agent: A Unified Remote Sensing Foundation Model for Multi-Platform and Multi-Modal Reasoning**|Arxiv2025|[RingMo-Agent](https://arxiv.org/abs/2507.20776)|null|
|**Geo-R1**|**Geo-R1: Improving Few-Shot Geospatial Referring Expression Understanding with Reinforcement Fine-Tuning**|Arxiv2025|[Geo-R1](https://arxiv.org/abs/2509.21976)|null|
|**RSThinker**|**Towards Faithful Reasoning in Remote Sensing: A Perceptually-Grounded GeoSpatial Chain-of-Thought for Vision-Language Models**|Arxiv2025|[RSThinker](https://arxiv.org/abs/2509.22221)|null|
|**FUSAR-KLIP**|**FUSAR-KLIP: Towards Multimodal Foundation Models for Remote Sensing**|Arxiv2025|[FUSAR-KLIP](https://arxiv.org/abs/2509.23927)|[link](https://github.com/yangyifremad/SARKnowLIP)|
|**GeoMag**|**GeoMag: A Vision-Language Model for Pixel-level Fine-Grained Remote Sensing Image Parsing**|ACMMM2025|[GeoMag](https://doi.org/10.1145/3746027.3754559)|null|
|**RemoteSAM**|**RemoteSAM: Towards Segment Anything for Earth Observation**|ACMMM2025|[RemoteSAM](https://arxiv.org/abs/2505.18022)|[link](https://github.com/1e12Leon/RemoteSAM)|
|**LRS-VQA**|**When Large Vision-Language Model Meets Large Remote Sensing Imagery: Coarse-to-Fine Text-Guided Token Pruning**|ICCV2025|[LRS-VQA](https://arxiv.org/abs/2503.07588)|[link](https://github.com/VisionXLab/LRS-VQA)|
|**UrbanLLaVA**|**UrbanLLaVA: A Multi-modal Large Language Model for Urban Intelligence with Spatial Reasoning and Understanding**|ICCV2025|[UrbanLLaVA](https://arxiv.org/abs/2506.23219)|[link](https://github.com/tsinghua-fib-lab/UrbanLLaVA)|
|**SARCLIP**|**SARCLIP: a multimodal foundation framework for SAR imagery via contrastive language-image pre-training**|ISPRS JPRS2025|[SARCLIP](https://doi.org/10.1016/j.isprsjprs.2025.10.011)|[link](https://huggingface.co/BiliSakura/SARCLIP-ViT-L-14)|
|**FUSE-RSVLM**|**FUSE-RSVLM: Feature Fusion Vision-Language Model for Remote Sensing**|Arxiv2025|[FUSE-RSVLM](https://arxiv.org/abs/2512.24022)|[link](https://github.com/Yunkaidang/RSVLM)|
|**Aquila**|**Aquila: A Hierarchically Aligned Vision-Language Model for Enhanced Remote Sensing Image Comprehension**|Journal of Remote Sensing 2026|[Aquila](https://doi.org/10.34133/remotesensing.1041)|null|
|**RSCoVLM**|**Co-Training Vision-Language Models for Remote Sensing Multi-Task Learning**|RS2026|[RSCoVLM](https://doi.org/10.3390/rs18020222)|[link](https://github.com/VisionXLab/RSCoVLM)|
|**GeoReason**|**GeoReason: Aligning Thinking And Answering In Remote Sensing Vision-Language Models Via Logical Consistency Reinforcement Learning**|Arxiv2026|[GeoReason](https://arxiv.org/abs/2601.04118)|[link](https://github.com/canlanqianyan/GeoReason)|
|**RemoteReasoner**|**RemoteReasoner: Towards Unifying Geospatial Reasoning Workflow**|AAAI2026|[RemoteReasoner](https://arxiv.org/abs/2507.19280)|null|
|**SkyMoE**|**SkyMoE: A Vision-Language Foundation Model for Enhancing Geospatial Interpretation with Mixture of Experts**|AAAI2026|[SkyMoE](https://arxiv.org/abs/2512.02517)|null|
|**GeoEyes**|**GeoEyes: On-Demand Visual Focusing for Evidence-Grounded Understanding of Ultra-High-Resolution Remote Sensing Imagery**|Arxiv2026|[GeoEyes](https://arxiv.org/abs/2602.14201)|[link](https://github.com/nanocm/GeoEyes)|
|**GeoSolver**|**GeoSolver: Scaling Test-Time Reasoning in Remote Sensing with Fine-Grained Process Supervision**|Arxiv2026|[GeoSolver](https://arxiv.org/abs/2603.09551)|null|
|**GeoAlignCLIP**|**GeoAlignCLIP: Enhancing Fine-Grained Vision-Language Alignment in Remote Sensing via Multi-Granular Consistency Learning**|Arxiv2026|[GeoAlignCLIP](https://arxiv.org/abs/2603.09566)|null|
|**RS-WorldModel**|**RS-WorldModel: a Unified Model for Remote Sensing Understanding and Future Sense Forecasting**|Arxiv2026|[RS-WorldModel](https://arxiv.org/abs/2603.14941)|null|
|**Decoding-the-Delta**|**Decoding the Delta: Unifying Remote Sensing Change Detection and Understanding with Multimodal Large Language Models**|Arxiv2026|[Paper](https://arxiv.org/abs/2604.14044)|null|
|**RemoteShield**|**RemoteShield: Enable Robust Multimodal Large Language Models for Earth Observation**|Arxiv2026|[RemoteShield](https://arxiv.org/abs/2604.17243)|null|
|**UniChange**|**UniChange: Unifying Change Detection with Multimodal Large Language Model**|CVPR2026|[UniChange](https://arxiv.org/abs/2511.02607)|[link](https://github.com/NKU-HLT/UniChange)|
|**ZoomEarth**|**ZoomEarth: Active Perception for Ultra-High-Resolution Geospatial Vision-Language Tasks**|CVPR2026|[ZoomEarth](https://arxiv.org/abs/2511.12267)|null|
|**UniGeoSeg**|**UniGeoSeg: Towards Unified Open-World Segmentation for Geospatial Scenes**|CVPR2026|[UniGeoSeg](https://arxiv.org/abs/2511.23332)|[link](https://github.com/MiliLab/UniGeoSeg)|
|**SegEarth-R2**|**SegEarth-R2: Towards Comprehensive Language-guided Segmentation for Remote Sensing Images**|CVPR2026|[SegEarth-R2](https://arxiv.org/abs/2512.20013)|[link](https://github.com/earth-insights/SegEarth-R2)|
|**FUSAR-GPT**|**FUSAR-GPT: A Spatiotemporal Feature-Embedded and Two-Stage Decoupled Visual Language Model for SAR Imagery**|CVPR2026|[FUSAR-GPT](https://arxiv.org/abs/2602.19190)|null|
|**SATtxt**|**Spectrally Distilled Representations Aligned with Instruction-Augmented LLMs for Satellite Imagery**|CVPR2026|[SATtxt](https://arxiv.org/abs/2602.22613)|[link](https://github.com/ikhado/sattxt)|
|**TerraScope**|**TerraScope: Pixel-Grounded Visual Reasoning for Earth Observation**|CVPR2026|[TerraScope](https://arxiv.org/abs/2603.19039)|null|
## Remote Sensing <ins>Generative</ins> Foundation Models

|Abbreviation|Title|Publication|Paper|Code & Weights|
|:---:|---|:---:|:---:|:---:|
|**GeoRSSD**|**RS5M: A Large Scale Vision-Language Dataset for Remote Sensing Vision-Language Foundation Model**|Arxiv2023|[Paper](https://arxiv.org/abs/2306.11300)|[link](https://huggingface.co/Zilun/GeoRSSD)|
|**-**|**Generate Your Own Scotland: Satellite Image Generation Conditioned on Maps**|NeurIPSW2023|[Paper](https://arxiv.org/abs/2308.16648)|[link](https://github.com/toastyfrosty/map-sat)|
|**CRS-Diff**|**CRS-Diff: Controllable Remote Sensing Image Generation with Diffusion Model**|Arxiv2024|[Paper](https://arxiv.org/abs/2403.11614)|[link](https://github.com/Sonettoo/CRS-Diff)|
|**DiffusionSat**|**DiffusionSat: A Generative Foundation Model for Satellite Imagery**|ICLR2024|[DiffusionSat](https://arxiv.org/abs/2312.03606)|[link](https://github.com/samar-khanna/DiffusionSat)|
|**HSIGene**|**HSIGene: A Foundation Model For Hyperspectral Image Generation**|Arxiv2024|[Paper](https://arxiv.org/abs/2409.12470)|[link](https://github.com/LiPang/HSIGene)|
|**Text2Earth**|**Text2Earth: Unlocking Text-driven Remote Sensing Image Generation with a Global-Scale Dataset and a Foundation Model**|GRSM2025|[Text2Earth](https://arxiv.org/abs/2501.00895)|[link](https://chen-yang-liu.github.io/Text2Earth/)|
|**MetaEarth**|**MetaEarth: A Generative Foundation Model for Global-Scale Remote Sensing Image Generation**|TPAMI2025|[MetaEarth](https://arxiv.org/abs/2405.13570)|[link](https://jiupinjia.github.io/metaearth/)|
|**EcoMapper**|**EcoMapper: Generative Modeling for Climate-Aware Satellite Imagery**|ICML2025|[EcoMapper](https://proceedings.mlr.press/v267/goktepe25a.html)|[link](https://github.com/maltevb/ecomapper)|
|**OSMGen**|**OSMGen: Highly Controllable Satellite Image Synthesis using OpenStreetMap Data**|NeurIPSW2025|[OSMGen](https://arxiv.org/abs/2511.00345)|[link](https://github.com/amir-zsh/OSMGen)|
|**Any2RSI**|**Any2RSI: Controllable Remote Sensing Text-to-Image Generation via Any Control and Enriched Description**|AAAI2026|[Any2RSI](https://ojs.aaai.org/index.php/AAAI/article/view/38283)|[link](https://github.com/lwCVer/RFD)|
|**GeoDiT**|**GeoDiT: Point-Conditioned Diffusion Transformer for Satellite Image Synthesis**|Arxiv2026|[GeoDiT](https://arxiv.org/abs/2603.02172)|null|
|**MetaEarth3D**|**MetaEarth3D: Unlocking World-scale 3D Generation with Spatially Scalable Generative Modeling**|Arxiv2026|[MetaEarth3D](https://arxiv.org/abs/2604.22828)|null|
## Remote Sensing <ins>Vision-Location</ins> Foundation Models

|Abbreviation|Title|Publication|Paper|Code & Weights|
|:---:|---|:---:|:---:|:---:|
|**CSP**|**CSP: Self-Supervised Contrastive Spatial Pre-Training for Geospatial-Visual Representations**|ICML2023|[CSP](https://arxiv.org/abs/2305.01118)|[link](https://gengchenmai.github.io/csp-website/)|
|**GeoCLIP**|**GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization**|NeurIPS2023|[GeoCLIP](https://arxiv.org/abs/2309.16020)|[link](https://vicentevivan.github.io/GeoCLIP/)|
|**SatCLIP**|**SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery**|AAAI2025|[SatCLIP](https://arxiv.org/abs/2311.17179)|[link](https://github.com/microsoft/satclip)|
|**GAIR**|**GAIR: Location-Aware Self-Supervised Contrastive Pre-Training with Geo-Aligned Implicit Representations**|Arxiv2025|[GAIR](https://arxiv.org/abs/2503.16683)|[link](https://github.com/zpl99/GAIR)|
|**RANGE**|**RANGE: Retrieval Augmented Neural Fields for Multi-Resolution Geo-Embeddings**|CVPR2025|[RANGE](https://arxiv.org/abs/2502.19781)|null|
|**Geo²**|**Geo²: Geometry-Guided Cross-view Geo-Localization and Image Synthesis**|CVPR2026|[Geo²](https://arxiv.org/abs/2603.25819)|null|
|**GeoBridge**|**GeoBridge: A Semantic-Anchored Multi-View Foundation Model Bridging Images and Text for Geo-Localization**|CVPR2026|[GeoBridge](https://arxiv.org/abs/2512.02697)|[link](https://github.com/MiliLab/GeoBridge)|
## Remote Sensing <ins>Vision-Audio</ins> Foundation Models

|Abbreviation|Title|Publication|Paper|Code & Weights|
|:---:|---|:---:|:---:|:---:|
|**-**|**Self-supervised audiovisual representation learning for remote sensing data**|JAG2022|[Paper](https://www.sciencedirect.com/science/article/pii/S1569843222003181)|[link](https://github.com/khdlr/SoundingEarth)|
|**GeoBind**|**GeoBind: Binding Text, Image, and Audio through Satellite Images**|IGARSS2024|[GeoBind](https://arxiv.org/abs/2404.11720)|null|
|**PSM**|**PSM: Learning Probabilistic Embeddings for Multi-scale Zero-Shot Soundscape Mapping**|ACM MM 2024|[PSM](https://arxiv.org/abs/2408.07050)|[link](https://github.com/mvrl/PSM)|
|**Sat2Sound**|**Sat2Sound: A Unified Framework for Zero-Shot Soundscape Mapping**|Arxiv2025|[Sat2Sound](https://arxiv.org/abs/2505.13777)|null|
## Remote Sensing Agents
|Abbreviation|Title|Publication|Paper|Code & Weights|
|:---:|---|:---:|:---:|:---:|
|**GeoLLM-QA**|**Evaluating Tool-Augmented Agents in Remote Sensing Platforms**|ICLR 2024 ML4RS Workshop|[Paper](https://arxiv.org/abs/2405.00709)|null|
|**GeoLLM-Engine**|**GeoLLM-Engine: A Realistic Environment for Building Geospatial Copilots.**|CVPRW2024|[Paper](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/html/Singh_GeoLLM-Engine_A_Realistic_Environment_for_Building_Geospatial_Copilots_CVPRW_2024_paper.html)|null|
|**RS-Agent**|**RS-Agent: Automating Remote Sensing Tasks through Intelligent Agent**|Arxiv2024|[Paper](https://arxiv.org/abs/2406.07089)|null|
|**Change-Agent**|**Change-Agent: Toward Interactive Comprehensive Remote Sensing Change Interpretation and Analysis**|TGRS2024|[Paper](https://ieeexplore.ieee.org/abstract/document/10591792)|[link](https://github.com/Chen-Yang-Liu/Change-Agent)|
|**GeoLLM-Squad**|**Multi-Agent Geospatial Copilots for Remote Sensing Workflows**|Arxiv2025|[Paper](https://arxiv.org/abs/2501.16254)|null|
|**-**|**Towards LLM Agents for Earth Observation: The UnivEARTH Dataset**|Arxiv2025|[Paper](https://arxiv.org/abs/2504.12110)|null|
|**AirSpatialBot**|**AirSpatialBot: A Spatially Aware Aerial Agent for Fine-Grained Vehicle Attribute Recognition and Retrieval**|IEEE TGRS2025|[Paper](https://doi.org/10.1109/tgrs.2025.3570895)|[link](https://github.com/VisionXLab/AirSpatialBot)|
|**ThinkGeo**|**ThinkGeo: Evaluating Tool-Augmented Agents for Remote Sensing Tasks**|Arxiv2025|[Paper](https://arxiv.org/abs/2505.23752)|[link](https://github.com/mbzuai-oryx/ThinkGeo)|
|**PEACE**|**PEACE: Empowering Geologic Map Holistic Understanding with MLLMs**|CVPR2025|[Paper](https://arxiv.org/abs/2501.06184)|[link](https://github.com/microsoft/PEACE?tab=readme-ov-file)|
|**Geo-OLM**|**Geo-OLM: Enabling Sustainable Earth Observation Studies with Cost-Efficient Open Language Models & State-Driven Workflows**|COMPASS'2025|[Paper](https://arxiv.org/abs/2504.04319)|[link](https://github.com/dstamoulis/geo-olms)|
|**REMSA**|**REMSA: Foundation Model Selection for Remote Sensing via a Constraint-Aware Agent**|Arxiv2025|[Paper](https://arxiv.org/abs/2511.17442)|[link](https://github.com/be-chen/REMSA)|
|**OpenEarthAgent**|**OpenEarthAgent: A Unified Framework for Tool-Augmented Geospatial Agents**|Arxiv2026|[Paper](https://arxiv.org/abs/2602.17665)|[link](https://github.com/mbzuai-oryx/OpenEarthAgent)|
|**OpenEarth-Agent**|**OpenEarth-Agent: From Tool Calling to Tool Creation for Open-Environment Earth Observation**|Arxiv2026|[Paper](https://arxiv.org/abs/2603.22148)|[link](https://github.com/walking-shadow/OpenEarth-Agent)|
|**Earth-Agent**|**Earth-Agent: Unlocking the Full Landscape of Earth Observation with Agents**|ICLR2026|[Paper](https://arxiv.org/abs/2509.23141)|[link](https://github.com/opendatalab/Earth-Agent)|
|**GeoMMAgent**|**GeoMMBench and GeoMMAgent: Toward Expert-Level Multimodal Intelligence in Geoscience and Remote Sensing**|CVPR2026|[Paper](https://arxiv.org/abs/2604.08896)|null|
|**IMAIA**|**IMAIA: Interactive Maps AI Assistant for Travel Planning and Geo-Spatial Intelligence**|CVPR2026|[Paper](https://arxiv.org/abs/2507.06993)|null|
## Benchmarks for RSFMs
|Abbreviation|Title|Publication|Paper|Link|Downstream Tasks|
|:---:|---|:---:|:---:|:---:|:---:|
|**-**|**Revisiting pre-trained remote sensing model benchmarks: resizing and normalization matters**|Arxiv2023|[Paper](https://arxiv.org/abs/2305.13456)|[link](https://github.com/isaaccorley/resize-is-all-you-need)|Classification|
|**GEO-Bench**|**GEO-Bench: Toward Foundation Models for Earth Monitoring**|Arxiv2023|[Paper](https://arxiv.org/abs/2306.03831)|[link](https://github.com/ServiceNow/geo-bench)|Classification & Segmentation|
|**FoMo-Bench**|**FoMo: Multi-Modal, Multi-Scale and Multi-Task Remote Sensing Foundation Models for Forest Monitoring**|Arxiv2023|[FoMo-Bench](https://arxiv.org/abs/2312.10114)|Coming soon|Classification & Segmentation & Detection for forest monitoring|
|**PhilEO**|**PhilEO Bench: Evaluating Geo-Spatial Foundation Models**|Arxiv2024|[Paper](https://arxiv.org/abs/2401.04464)|[link](https://github.com/91097luke/phileo-bench)|Segmentation & Regression estimation|
|**SkySense**|**SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery**|CVPR2024|[SkySense](https://arxiv.org/abs/2312.10115)|Targeted open-source|Classification & Segmentation & Detection & Change detection & Multi-Modal Segmentation: Time-insensitive LandCover Mapping & Multi-Modal Segmentation: Time-sensitive Crop Mapping & Multi-Modal Scene Classification|
|**VLEO-Bench**|**Good at captioning, bad at counting: Benchmarking GPT-4V on Earth observation data**|Arxiv2024|[VLEO-bench](https://arxiv.org/abs/2401.17600)|[link](https://vleo.danielz.ch/)| Location Recognition & Captioning & Scene Classification & Counting & Detection & Change detection|
|**VRSBench**|**VRSBench: A Versatile Vision-Language Benchmark Dataset for Remote Sensing Image Understanding**|NeurIPS2024|[VRSBench](https://arxiv.org/abs/2406.12384)|[link](https://vrsbench.github.io/)|Image Captioning & Object Referring & Visual Question Answering|
|**UrBench**|**UrBench: A Comprehensive Benchmark for Evaluating Large Multimodal Models in Multi-View Urban Scenarios**|AAAI2025|[UrBench](https://doi.org/10.1609/aaai.v39i10.33163)|[link](https://opendatalab.github.io/UrBench/)|Object Referring & Visual Question Answering & Counting & Scene Classification & Location Recognition & Geolocalization|
|**PANGAEA**|**PANGAEA: A Global and Inclusive Benchmark for Geospatial Foundation Models**|Arxiv2024|[PANGAEA](https://arxiv.org/abs/2412.04204)|[link](https://github.com/yurujaja/pangaea-bench)|Segmentation & Change detection & Regression|
|**CHOICE**|**CHOICE: Evaluating and Understanding Vision-Language Model Choices in Remote Sensing**|NeurIPS2025|[CHOICE](https://neurips.cc/virtual/2025/poster/121749)|[link](https://github.com/ShawnAn-WHU/CHOICE)|Perception & Reasoning|
|**GEO-Bench-VLM**|**GEO-Bench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks**|ICCV2025|[GEO-Bench-VLM](https://arxiv.org/abs/2411.19325)|[link](https://github.com/The-AI-Alliance/GEO-Bench-VLM)|Scene Understanding & Counting & Object Classification & Event Detection & Spatial Relations|
|**Copernicus-Bench**|**Towards a Unified Copernicus Foundation Model for Earth Vision**|ICCV2025|[Copernicus-Bench](https://arxiv.org/abs/2503.11849)|[link](https://github.com/zhu-xlab/Copernicus-FM)|Segmentation & Classification & Change detection & Regression|
|**REOBench**|**REOBench: Benchmarking Robustness of Earth Observation Foundation Models**|Arxiv2025|[REOBench](https://arxiv.org/abs/2505.16793)|[link](https://github.com/lx709/REOBench)|Robustness across 6 Earth observation tasks|
|**Plantation Bench**|**Plantation Bench: A Multiscale, Multimodal Remote Sensing Benchmark for Plantation Mapping Under Distribution Shift**|ICCVW2025|[Plantation Bench](https://doi.org/10.1109/iccvw69036.2025.00310)|null|Plantation Mapping under Distribution Shift|
|**ChatEarthBench**|**ChatEarthBench: Benchmarking multimodal large language models for Earth observation**|IEEE GRSM2026|[ChatEarthBench](https://doi.org/10.1109/mgrs.2026.3650840)|null|Benchmarking EO multimodal large language models|
|**GeoReason-Bench**|**GeoReason: Aligning Thinking And Answering In Remote Sensing Vision-Language Models Via Logical Consistency Reinforcement Learning**|Arxiv2026|[GeoReason-Bench](https://arxiv.org/abs/2601.04118)|[link](https://github.com/canlanqianyan/GeoReason)|Logical consistency & multi-step reasoning|
|**Earth-Bench**|**Earth-Agent: Unlocking the Full Landscape of Earth Observation with Agents (Earth-Bench is the benchmark introduced in this paper)**|ICLR2026|[Earth-Agent](https://arxiv.org/abs/2509.23141)|[link](https://huggingface.co/datasets/Sssunset/Earth-Bench)|Tool-augmented EO reasoning & multi-step planning & quantitative spatiotemporal analysis|
|**OmniEarth**|**OmniEarth: A Benchmark for Evaluating Vision-Language Models in Geospatial Tasks**|Arxiv2026|[OmniEarth](https://arxiv.org/abs/2603.09471)|[link](https://huggingface.co/datasets/sjeeudd/OmniEarth)|Perception & Reasoning & Robustness across geospatial tasks|
|**SpatialSky-Bench**|**Is your VLM Sky-Ready? A Comprehensive Spatial Intelligence Benchmark for UAV Navigation**|CVPR2026|[SpatialSky-Bench](https://arxiv.org/abs/2511.13269)|null|UAV spatial intelligence evaluation for Vision-Language Models|
|**GeoMMBench**|**GeoMMBench and GeoMMAgent: Toward Expert-Level Multimodal Intelligence in Geoscience and Remote Sensing**|CVPR2026|[GeoMMBench](https://arxiv.org/abs/2604.08896)|null|Expert-level multimodal QA across RS disciplines, sensors & tasks|
|**RSVLM-QA**|**RSVLM-QA: A Benchmark Dataset for Remote Sensing Vision Language Model-based Question Answering**|ACM MM2025|[RSVLM-QA](https://arxiv.org/abs/2508.07918)|[link](https://github.com/StarZi0213/RSVLM-QA)|Visual Question Answering & Image Captioning & Counting|
|**Geo3DVQA**|**Geo3DVQA: Evaluating Vision-Language Models for 3D Geospatial Reasoning from Aerial Imagery**|WACV2026|[Geo3DVQA](https://arxiv.org/abs/2512.07276)|[link](https://github.com/mm1129/Geo3DVQA)|3D geospatial reasoning & height-aware spatial analysis|
|**OpenEarth-Bench**|**OpenEarth-Agent: From Tool Calling to Tool Creation for Open-Environment Earth Observation (OpenEarth-Bench is the benchmark introduced in this paper)**|Arxiv2026|[OpenEarth-Agent](https://arxiv.org/abs/2603.22148)|[link](https://github.com/walking-shadow/OpenEarth-Agent)|Full-pipeline EO across 7 application domains|
|**GeoAgentBench**|**GeoAgentBench: A Dynamic Execution Benchmark for Tool-Augmented Agents in Spatial Analysis**|Arxiv2026|[GeoAgentBench](https://arxiv.org/abs/2604.13888)|null|Dynamic execution evaluation for tool-augmented GIS agents|


## (Large-scale) Pre-training Datasets

|Abbreviation|Title|Publication|Paper|Attribute|Link|
|:---:|---|:---:|:---:|:---:|:---:|
|**fMoW**|**Functional Map of the World**|CVPR2018|[fMoW](https://openaccess.thecvf.com/content_cvpr_2018/html/Christie_Functional_Map_of_CVPR_2018_paper.html)|**Vision**|[link](https://github.com/fMoW)|
|**SEN12MS**|**SEN12MS -- A Curated Dataset of Georeferenced Multi-Spectral Sentinel-1/2 Imagery for Deep Learning and Data Fusion**|-|[SEN12MS](https://arxiv.org/abs/1906.07789)|**Vision**|[link](https://mediatum.ub.tum.de/1474000)|
|**BEN-MM**|**BigEarthNet-MM: A Large Scale Multi-Modal Multi-Label Benchmark Archive for Remote Sensing Image Classification and Retrieval**|GRSM2021|[BEN-MM](https://ieeexplore.ieee.org/abstract/document/9552024)|**Vision**|[link](https://bigearth.net/)|
|**MillionAID**|**On Creating Benchmark Dataset for Aerial Image Interpretation: Reviews, Guidances, and Million-AID**|JSTARS2021|[MillionAID](https://ieeexplore.ieee.org/abstract/document/9393553)|**Vision**|[link](https://captain-whu.github.io/DiRS/)|
|**SeCo**|**Seasonal Contrast: Unsupervised Pre-Training From Uncurated Remote Sensing Data**|ICCV2021|[SeCo](https://openaccess.thecvf.com/content/ICCV2021/html/Manas_Seasonal_Contrast_Unsupervised_Pre-Training_From_Uncurated_Remote_Sensing_Data_ICCV_2021_paper.html)|**Vision**|[link](https://github.com/ServiceNow/seasonal-contrast)|
|**fMoW-S2**|**SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery**|NeurIPS2022|[fMoW-S2](https://proceedings.neurips.cc/paper_files/paper/2022/hash/01c561df365429f33fcd7a7faa44c985-Abstract-Conference.html)|**Vision**|[link](https://purl.stanford.edu/vg497cb6002)|
|**TOV-RS-Balanced**|**TOV: The original vision model for optical remote sensing image understanding via self-supervised learning**|JSTARS2023|[TOV](https://ieeexplore.ieee.org/abstract/document/10110958)|**Vision**|[link](https://github.com/GeoX-Lab/G-RSIM/tree/main/TOV_v1)|
|**SSL4EO-S12**|**SSL4EO-S12: A Large-Scale Multi-Modal, Multi-Temporal Dataset for Self-Supervised Learning in Earth Observation**|GRSM2023|[SSL4EO-S12](https://arxiv.org/abs/2211.07044)|**Vision**|[link](https://github.com/zhu-xlab/SSL4EO-S12)|
|**SSL4EO-L**|**SSL4EO-L: Datasets and Foundation Models for Landsat Imagery**|Arxiv2023|[SSL4EO-L](https://arxiv.org/abs/2306.09424)|**Vision**|[link](https://github.com/microsoft/torchgeo)|
|**SatlasPretrain**|**SatlasPretrain: A Large-Scale Dataset for Remote Sensing Image Understanding**|ICCV2023|[SatlasPretrain](https://arxiv.org/abs/2211.15660)|**Vision (Supervised)**|[link](https://github.com/allenai/satlas/blob/main/SatlasPretrain.md)|
|**CACo**|**Change-Aware Sampling and Contrastive Learning for Satellite Images**|CVPR2023|[CACo](https://openaccess.thecvf.com/content/CVPR2023/html/Mall_Change-Aware_Sampling_and_Contrastive_Learning_for_Satellite_Images_CVPR_2023_paper.html)|**Vision**|[Coming soon](https://github.com/utkarshmall13/CACo)|
|**SAMRS**|**SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model**|NeurIPS2023|[SAMRS](https://arxiv.org/abs/2305.02034)|**Vision**|[link](https://github.com/ViTAE-Transformer/SAMRS)|
|**RSVG**|**RSVG: Exploring Data and Models for Visual Grounding on Remote Sensing Data**|TGRS2023|[RSVG](https://ieeexplore.ieee.org/document/10056343)|**Vision-Language**|[link](https://github.com/ZhanYang-nwpu/RSVG-pytorch)|
|**RS5M**|**RS5M: A Large Scale Vision-Language Dataset for Remote Sensing Vision-Language Foundation Model**|Arxiv2023|[RS5M](https://arxiv.org/abs/2306.11300)|**Vision-Language**|[link](https://github.com/om-ai-lab/RS5M)|
|**GEO-Bench**|**GEO-Bench: Toward Foundation Models for Earth Monitoring**|Arxiv2023|[GEO-Bench](https://arxiv.org/abs/2306.03831)|**Vision (Evaluation)**|[link](https://github.com/ServiceNow/geo-bench)|
|**RSICap & RSIEval**|**RSGPT: A Remote Sensing Vision Language Model and Benchmark**|Arxiv2023|[RSGPT](https://arxiv.org/abs/2307.15266)|**Vision-Language**|[Coming soon](https://github.com/Lavender105/RSGPT)|
|**Clay**|**Clay Foundation Model**|-|null|**Vision**|[link](https://clay-foundation.github.io/model/)|
|**SATIN**|**SATIN: A Multi-Task Metadataset for Classifying Satellite Imagery using Vision-Language Models**|ICCVW2023|[SATIN](https://arxiv.org/abs/2304.11619)|**Vision-Language**|[link](https://satinbenchmark.github.io/)|
|**SkyScript**|**SkyScript: A Large and Semantically Diverse Vision-Language Dataset for Remote Sensing**|AAAI2024|[SkyScript](https://arxiv.org/abs/2312.12856)|**Vision-Language**|[link](https://github.com/wangzhecheng/SkyScript)|
|**ChatEarthNet**|**ChatEarthNet: a global-scale image-text dataset empowering vision-language geo-foundation models**|ESSD2025|[ChatEarthNet](https://doi.org/10.5194/essd-17-1245-2025)|**Vision-Language**|[link](https://github.com/zhu-xlab/ChatEarthNet)|
|**LuoJiaHOG**|**LuoJiaHOG: A hierarchy oriented geo-aware image caption dataset for remote sensing image-text retrieval**|ISPRS JPRS2025|[LuoJiaHOG](https://doi.org/10.1016/j.isprsjprs.2025.02.009)|**Vision-Language**|null|
|**MMEarth**|**MMEarth: Exploring Multi-Modal Pretext Tasks For Geospatial Representation Learning**|Arxiv2024|[MMEarth](https://arxiv.org/abs/2405.02771)|**Vision**|[link](https://vishalned.github.io/mmearth/)|
|**SeeFar**|**SeeFar: Satellite Agnostic Multi-Resolution Dataset for Geospatial Foundation Models**|Arxiv2024|[SeeFar](https://arxiv.org/abs/2406.06776)|**Vision**|[link](https://coastalcarbon.ai/seefar)|
|**FIT-RS**|**SkySenseGPT: A Fine-Grained Instruction Tuning Dataset and Model for Remote Sensing Vision-Language Understanding**|Arxiv2024|[Paper](https://arxiv.org/abs/2406.10100)|**Vision-Language**|[link](https://github.com/Luo-Z13/SkySenseGPT)|
|**RS-GPT4V**|**RS-GPT4V: A Unified Multimodal Instruction-Following Dataset for Remote Sensing Image Understanding**|Arxiv2024|[Paper](https://arxiv.org/abs/2406.12479)|**Vision-Language**|[link](https://github.com/GeoX-Lab/RS-GPT4V/tree/main)|
|**RS-4M**|**Harnessing Massive Satellite Imagery with Efficient Masked Image Modeling**|ICCV2025|[RS-4M](https://arxiv.org/abs/2406.11933)|**Vision**|[link](https://github.com/Fengxiang23/SelectiveMAE)|
|**Major TOM**|**Major TOM: Expandable Datasets for Earth Observation**|Arxiv2024|[Major TOM](https://arxiv.org/abs/2402.12095)|**Vision**|[link](https://huggingface.co/Major-TOM)|
|**VRSBench**|**VRSBench: A Versatile Vision-Language Benchmark Dataset for Remote Sensing Image Understanding**|NeurIPS2024|[VRSBench](https://arxiv.org/abs/2406.12384)|**Vision-Language**|[link](https://vrsbench.github.io/)|
|**MMM-RS**|**MMM-RS: A Multi-modal, Multi-GSD, Multi-scene Remote Sensing Dataset and Benchmark for Text-to-Image Generation**|Arxiv2024|[MMM-RS](https://arxiv.org/abs/2410.22362)|**Vision-Language**|[link](https://github.com/ljl5261/MMM-RS)|
|**DDFAV**|**DDFAV: Remote Sensing Large Vision Language Models Dataset and Evaluation Benchmark**|RS2025|[DDFAV](https://doi.org/10.3390/rs17040719)|**Vision-Language**|[link](https://github.com/HaodongLi2024/rspope)|
|**M3LEO**|**A Multi-Modal, Multi-Label Earth Observation Dataset Integrating Interferometric SAR and Multispectral Data**|NeurIPS2024|[M3LEO](https://proceedings.neurips.cc/paper_files/paper/2024/file/bd194b579f60879e04ca9ce8a4ea5da1-Paper-Datasets_and_Benchmarks_Track.pdf)|**Vision**|[link](https://github.com/spaceml-org/M3LEO)|
|**Copernicus-Pretrain**|**Towards a Unified Copernicus Foundation Model for Earth Vision**|ICCV2025|[Copernicus-Pretrain](https://arxiv.org/abs/2503.11849)|**Vision**|[link](https://github.com/zhu-xlab/Copernicus-FM)|
|**DGTRSD**|**DGTRSD & DGTRS-CLIP: A Dual-Granularity Remote Sensing Image-Text Dataset and Vision Language Foundation Model for Alignment**|Arxiv2025|[Paper](https://arxiv.org/abs/2503.19311)|**Vision-Language**|[link](https://github.com/MitsuiChen14/DGTRS)|
|**EarthDial-Instruct**|**EarthDial: Turning Multi-sensory Earth Observations to Interactive Dialogues**|CVPR2025|[Paper](https://arxiv.org/abs/2412.15190)|**Vision-Language**|[link](https://github.com/hiyamdebary/EarthDial)|
|**GeoPixelD**|**GeoPixel: Pixel Grounding Large Multimodal Model in Remote Sensing**|ICML2025|[Paper](https://arxiv.org/abs/2501.13925)|**Vision-Language**|[link](https://github.com/mbzuai-oryx/GeoPixel)|
|**GeoPixInstruct**|**GeoPix: Multi-Modal Large Language Model for Pixel-level Image Understanding in Remote Sensing**|IEEE GRSM2025|[Paper](https://arxiv.org/abs/2501.06828)|**Vision-Language**|[link](https://github.com/Norman-Ou/GeoPix)|
|**GeoLangBind-2M**|**Rethinking Remote Sensing CLIP: Leveraging Multimodal Large Language Models for High-Quality Vision-Language Dataset**|ICONIP2024|[Paper](https://doi.org/10.1007/978-981-96-6972-1_29)|**Vision-Language**|[link](https://github.com/xiong-zhitong/GeoLB-SigLIP)|
|**Falcon_SFT**|**Falcon: A Remote Sensing Vision-Language Foundation Model**|Arxiv2025|[Paper](https://arxiv.org/abs/2503.11070)|**Vision-Language**|[link](https://github.com/TianHuiLab/Falcon)|
|**UnivEARTH**|**Towards LLM Agents for Earth Observation: The UnivEARTH Dataset**|Arxiv2025|[Paper](https://arxiv.org/abs/2504.12110)|**Vision-Language & Agents**|null|
|**RemoteSAM-270K**|**RemoteSAM: Towards Segment Anything for Earth Observation**|ACMMM2025|[Paper](https://arxiv.org/abs/2505.18022)|**Vision-Language**|[link](https://github.com/1e12Leon/RemoteSAM)|
|**OpenEarthAgent Dataset**|**OpenEarthAgent: A Unified Framework for Tool-Augmented Geospatial Agents**|Arxiv2026|[Paper](https://arxiv.org/abs/2602.17665)|**Vision-Language & Agents**|[link](https://github.com/mbzuai-oryx/OpenEarthAgent)|
|**UHR-CoZ**|**GeoEyes: On-Demand Visual Focusing for Evidence-Grounded Understanding of Ultra-High-Resolution Remote Sensing Imagery**|Arxiv2026|[Paper](https://arxiv.org/abs/2602.14201)|**Vision-Language**|[link](https://github.com/nanocm/GeoEyes)|
|**SOMA-1M**|**SOMA-1M: A Large-Scale SAR-Optical Multi-resolution Alignment Dataset for Multi-Task Remote Sensing**|Arxiv2026|[SOMA-1M](https://arxiv.org/abs/2602.05480)|**Vision (SAR-Optical)**|[link](https://github.com/PeihaoWu/SOMA-1M)|

## Embeddings data

|Abbreviation|Title|Publication|Paper|Code|Dataset / Product|
|:---:|---|:---:|:---:|:---:|:---:|
|**CLAY Embeddings**|**Clay Model v0 Embeddings**|Source Cooperative2024|null|[link](https://github.com/Clay-foundation)|[link](https://source.coop/clay/clay-model-v0-embeddings)|
|**Major TOM Embeddings**|**Global and Dense Embeddings of Earth: Major TOM Floating in the Latent Space**|Arxiv2024|[Paper](https://arxiv.org/abs/2412.05600)|[link](https://github.com/ESA-PhiLab/Major-TOM)|[link](https://huggingface.co/Major-TOM)|
|**Earth Genome Embeddings**|**Embeddings for all**|Medium2025|[Paper](https://medium.com/earthrisemedia/embeddings-for-all-0e0a29415b26)|null|[link](https://source.coop/earthgenome/earthindexembeddings)|
|**TESSERA**|**TESSERA: Temporal Embeddings of Surface Spectra for Earth Representation and Analysis**|CVPR2026|[Paper](https://arxiv.org/abs/2506.20380)|[link](https://github.com/ucam-eo/tessera)|[link](https://github.com/ucam-eo/geotessera)|
|**AlphaEarth**|**AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data**|Arxiv2025|[Paper](https://arxiv.org/abs/2507.22291)|null|[link](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)|
|**ESD**|**Democratizing planetary-scale analysis: An ultra-lightweight Earth embedding database for accurate and flexible global land monitoring**|Arxiv2026|[Paper](https://arxiv.org/abs/2601.11183)|[link](https://github.com/shuangchencc/ESD)|[link](https://data-starcloud.pcl.ac.cn/iearthdata/64)|
|**Copernicus-Embed**|**Towards a Unified Copernicus Foundation Model for Earth Vision**|ICCV2025 Oral|[Paper](https://arxiv.org/abs/2503.11849)|[link](https://github.com/zhu-xlab/Copernicus-FM)|[link](https://huggingface.co/datasets/wangyi111/Copernicus-Embed-025deg)|

## Relevant Projects
|Title|Link|Brief Introduction|
|---|:---:|:---:|
|**RSFMs (Remote Sensing Foundation Models) Playground**|[link](https://github.com/synativ/RSFMs)|An open-source playground to streamline the evaluation and fine-tuning of RSFMs on various datasets.|
|**PANGAEA**|[link](https://github.com/yurujaja/pangaea-bench)|A Global and Inclusive Benchmark for Geospatial Foundation Models.|
|**GeoFM**|[link](https://github.com/xiong-zhitong/GeoFM)|Evaluation of Foundation Models for Earth Observation.|
|**rs-embed**|[link](https://github.com/cybergis/rs-embed)|One line code to get Any Remote Sensing Foundation Model (RSFM) embeddings for Any Place and Any Time.|
|**TerraTorch**|[link](https://github.com/terrastackai/terratorch)|A PyTorch toolkit for fine-tuning Geospatial Foundation Models, supporting Prithvi, TerraMind, SatMAE, ScaleMAE, DOFA, Clay, and more.|
|**TorchGeo**|[link](https://github.com/microsoft/torchgeo)|A PyTorch domain library for geospatial data, providing datasets, samplers, transforms, and pre-trained models.|
|**Awesome-Geospatial-Embeddings**|[link](https://github.com/hfangcat/Awesome-Geospatial-Embeddings)|A curated list of papers on how to represent Earth data in embedding space — spatial, temporal, or semantic.|

## Survey/Commentary Papers
|Title|Publication|Paper|Attribute|
|---|:---:|:---:|:---:|
|**The Potential of Visual ChatGPT For Remote Sensing**|Arxiv2023|[Paper](https://arxiv.org/abs/2304.13009)|**Vision-Language**|
|**Self-Supervised Remote Sensing Feature Learning: Learning Paradigms, Challenges, and Future Works**|TGRS2023|[Paper](https://ieeexplore.ieee.org/abstract/document/10126079)|**Vision & Vision-Language**|
|**Revisiting pre-trained remote sensing model benchmarks: resizing and normalization matters**|Arxiv2023|[Paper](https://arxiv.org/abs/2305.13456)|**Vision**|
|**An Agenda for Multimodal Foundation Models for Earth Observation**|IGARSS2023|[Paper](https://ieeexplore.ieee.org/abstract/document/10282966)|**Vision**|
|**遥感大模型：进展与前瞻**|武汉大学学报 (信息科学版) 2023|[Paper](http://ch.whu.edu.cn/cn/article/doi/10.13203/j.whugis20230341?viewType=HTML)|**Vision & Vision-Language**|
|**地理人工智能样本：模型、质量与服务**|武汉大学学报 (信息科学版) 2023|[Paper](http://ch.whu.edu.cn/article/id/5e67ed6a-aae5-4ec0-ad1b-f2aba89f4617)|**-**|
|**Brain-Inspired Remote Sensing Foundation Models and Open Problems: A Comprehensive Survey**|JSTARS2023|[Paper](https://ieeexplore.ieee.org/abstract/document/10254282)|**Vision & Vision-Language**|
|**遥感基础模型发展综述与未来设想**|遥感学报2023|[Paper](https://www.ygxb.ac.cn/zh/article/doi/10.11834/jrs.20233313/)|**-**|
|**On the Promises and Challenges of Multimodal Foundation Models for Geographical, Environmental, Agricultural, and Urban Planning Applications**|Arxiv2023|[Paper](https://arxiv.org/abs/2312.17016)|**Vision-Language**|
|**Transfer learning in environmental remote sensing**|RSE2024|[Paper](https://www.sciencedirect.com/science/article/pii/S0034425723004765)|**Transfer learning**|
|**Vision-Language Models in Remote Sensing: Current Progress and Future Trends**|IEEE GRSM2024|[Paper](https://arxiv.org/abs/2305.05726)|**Vision-Language**|
|**On the Foundations of Earth and Climate Foundation Models**|Arxiv2024|[Paper](https://arxiv.org/abs/2405.04285)|**Vision & Vision-Language**|
|**多模态遥感基础大模型：研究现状与未来展望**|测绘学报2024|[Paper](http://xb.chinasmp.com/CN/10.11947/j.AGCS.2024.20240019.)|**Vision & Vision-Language & Generative & Vision-Location**|
|**Towards Vision-Language Geo-Foundation Model: A Survey**|Arxiv2024|[Paper](https://arxiv.org/abs/2406.09385)|**Vision-Language**|
|**Vision Foundation Models in Remote Sensing: A Survey**|Arxiv2024|[Paper](https://arxiv.org/abs/2408.03464)|**Vision**|
|**Foundation model for generalist remote sensing intelligence: Potentials and prospects**|Science Bulletin2024|[Paper](https://www.sciencedirect.com/science/article/pii/S2095927324006510?via%3Dihub)|**-**|
|**Advancements in Visual Language Models for Remote Sensing: Datasets, Capabilities, and Enhancement Techniques**|Arxiv2024|[Paper](https://arxiv.org/abs/2410.17283)|**Vision-Language**|
|**When Geoscience Meets Foundation Models: Toward a general geoscience artificial intelligence system**|IEEE GRSM2024|[Paper](https://ieeexplore.ieee.org/abstract/document/10770814)|**Vision & Vision-Language**|
|**Towards the next generation of Geospatial Artificial Intelligence**|JAG2025|[Paper](https://www.sciencedirect.com/science/article/pii/S1569843225000159)|**-**|
|**When Remote Sensing Meets Foundation Model: A Survey and Beyond**|RS2025|[Paper](https://doi.org/10.3390/rs17020179)|**Vision & Vision-Language & Generative & Agents**|
|**Vision Foundation Models in Remote Sensing: A survey**|IEEE GRSM2025|[Paper](https://ieeexplore.ieee.org/abstract/document/10916803)|**Vision**|
|**Remote Sensing Tuning: A Survey**|CVM2025|[Paper](https://doi.org/10.26599/cvm.2025.9450490)|**Vision & Vision-Language**|
|**Unleashing the potential of remote sensing foundation models via bridging data and computility islands**|The Innovation2025|[Paper](https://www.cell.com/the-innovation/fulltext/S2666-6758(25)00044-X)|**-**|
|**A Survey on Remote Sensing Foundation Models: From Vision to Multimodality**|Arxiv2025|[Paper](https://arxiv.org/abs/2503.22081)|**-**|
|**Foundation Models for Remote Sensing and Earth Observation: A survey**|IEEE GRSM2025|[Paper](https://doi.org/10.1109/mgrs.2025.3576766)|**Vision & Vision-Language**|
|**Vision-Language Modeling Meets Remote Sensing: Models, datasets, and perspectives**|IEEE GRSM2025|[Paper](https://doi.org/10.1109/mgrs.2025.3572702)|**Vision-Language**|
|**MIMRS: A Survey on Masked Image Modeling in Remote Sensing**|IGARSS2025|[Paper](https://doi.org/10.1109/igarss55030.2025.11243448)|**Vision**|
|**A Review of Challenges and Applications in Remote Sensing Foundation Models**|IGARSS2025|[Paper](https://doi.org/10.1109/igarss55030.2025.11242732)|**Vision & Vision-Language**|
|**On the Status of Foundation Models for SAR Imagery**|Arxiv2025|[Paper](https://arxiv.org/abs/2509.21722)|**Vision (SAR)**|
|**Advances on Multimodal Remote Sensing Foundation Models for Earth Observation Downstream Tasks: A Survey**|RS2025|[Paper](https://doi.org/10.3390/rs17213532)|**Vision & Vision-Language**|
|**Agentic AI in Remote Sensing: Foundations, Taxonomy, and Emerging Systems**|WACVW2026|[Paper](https://arxiv.org/abs/2601.01891)|**Agents**|
|**Onboard Deployment of Remote Sensing Foundation Models: A Comprehensive Review of Architecture, Optimization, and Hardware**|RS2026|[Paper](https://doi.org/10.3390/rs18020298)|**Vision & Vision-Language**|
|**On the foundations of Earth foundation models**|Communications Earth & Environment 2026|[Paper](https://doi.org/10.1038/s43247-025-03127-x)|**Vision & Vision-Language**|
|**A Genealogy of Foundation Models in Remote Sensing**|ACM TSAS2026|[Paper](https://doi.org/10.1145/3789505)|**Vision & Vision-Language**|
|**Foundation Models in Remote Sensing: Evolving from Unimodality to Multimodality**|IEEE GRSM2026|[Paper](https://arxiv.org/abs/2603.00988)|**Vision & Vision-Language**|
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

@article{li2025unleashing,
  title={Unleashing the potential of remote sensing foundation models via bridging data and computility islands},
  author={Li, Yansheng and Tan, Jieyi and Dang, Bo and Ye, Mang and Bartalev, Sergey A and Shinkarenko, Stanislav and Wang, Linlin and Zhang, Yingying and Ru, Lixiang and Guo, Xin and others},
  journal={The Innovation},
  year={2025},
  publisher={Elsevier}
}

@article{wu2025semantic,
  author = {Wu, Kang and Zhang, Yingying and Ru, Lixiang and Dang, Bo and Lao, Jiangwei and Yu, Lei and Luo, Junwei and Zhu, Zifan and Sun, Yue and Zhang, Jiahao and Zhu, Qi and Wang, Jian and Yang, Ming and Chen, Jingdong and Zhang, Yongjun and Li, Yansheng},
  title= {A semantic‑enhanced multi‑modal remote sensing foundation model for Earth observation},
  journal= {Nature Machine Intelligence},
  year= {2025},
  doi= {10.1038/s42256-025-01078-8},
  url= {https://doi.org/10.1038/s42256-025-01078-8}
}

@inproceedings{zhu2025skysense,
  title={Skysense-o: Towards open-world remote sensing interpretation with vision-centric visual-language modeling},
  author={Zhu, Qi and Lao, Jiangwei and Ji, Deyi and Luo, Junwei and Wu, Kang and Zhang, Yingying and Ru, Lixiang and Wang, Jian and Chen, Jingdong and Yang, Ming and others},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={14733--14744},
  year={2025}
}

@article{luo2024skysensegpt,
  title={Skysensegpt: A fine-grained instruction tuning dataset and model for remote sensing vision-language understanding},
  author={Luo, Junwei and Pang, Zhen and Zhang, Yongjun and Wang, Tingzhu and Wang, Linlin and Dang, Bo and Lao, Jiangwei and Wang, Jian and Chen, Jingdong and Tan, Yihua and others},
  journal={arXiv preprint arXiv:2406.10100},
  year={2024}
}
```
