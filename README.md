# PlaLLM

## Learning Semantic Plasticity Relationship via Hierarchical Correlational Modeling for Robust Head Pose Estimation

## Abstract

Head pose estimation (HPE) is important for human–robot interaction, yet occlusion, extreme head angles, and complex lighting remain challenging in practical settings. We propose **PlaHPE**, a plasticity relationship learning framework for robust HPE through hierarchical correlational modeling. The framework builds on two insights: facial features are correlated across different levels, and head rotation changes the relative importance of facial regions. A **Semantic Plasticity Generation (SPG)** module uses large language models to describe facial plasticity at multiple granularities. A dual-branch architecture then combines a **Plasticity Relationship Mining (PRM)** module, which aligns textual and visual features to learn high-level semantic knowledge, with a **Plasticity Feature Interaction (PFI)** module, which aggregates multigranularity features for pose prediction. Experiments on AFLW2000 and BIWI demonstrate improved estimation accuracy and robustness. At inference time, prediction uses only the vision encoding branch, without additional textual input.

## Main Contributions

1. **Semantic plasticity relationship learning.** We model facial changes induced by head rotation to identify the relative importance of different facial regions and adaptively focus on informative features across poses and environments.
2. **Hierarchical semantic guidance.** SPG generates descriptions at the **keypoint, region, and facial levels**, while PRM aligns these descriptions with visual features to learn rotation-dependent plasticity relationships.
3. **A dual-branch architecture with multigranularity interaction.** The plasticity guidance branch supports visual representation learning during training, and PFI aggregates key features across scales. Inference relies on the vision encoding branch alone.
4. **Evaluation on standard benchmarks.** Experiments on AFLW2000 and BIWI, with training on 300W-LP, demonstrate the effectiveness of the framework. The ViT-B variant achieves MAEs of **3.45°** and **3.29°**, respectively.

## Method

![Architecture of PlaHPE, showing the vision encoding branch, plasticity guidance branch, SPG, PRM, and PFI.](assets/framework.png)

*PlaHPE framework overview.*


## Datasets

The paper trains on **300W-LP** and evaluates on **AFLW2000** and **BIWI**. AFLW2000 refers to the **AFLW2000-3D** benchmark provided by the dataset authors.

| Dataset | Use in the paper | Description | Official source / download |
| --- | --- | --- | --- |
| **300W-LP** | Training | Approximately 61K synthesized face images with large pose variations, expanded to approximately 122K with flipping. | [download](https://drive.google.com/file/d/0B7OEHD3T4eCkVGs0TkhUWFN6N1k/view?usp=sharing) |
| **AFLW2000-3D** | Evaluation | 2,000 in-the-wild images with fitted 3D faces and 68 facial landmarks. | [download](https://drive.google.com/file/d/0B7OEHD3T4eCkZXU4Zy1zZVgzUHc/view?resourcekey=0-h0GyCbYRA67dgDbelEFxhA&usp=sharing) |

Both download links above are provided by Xiangyu Zhu, an author of **Face Alignment Across Large Poses: A 3D Solution (CVPR 2016)**, in the corresponding publication entry on his [official homepage](https://xiangyuzhu-open.github.io/homepage/#selected-publications). The links point to the original author-provided archives, rather than third-party mirrors. See also the [original 3DDFA project page](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm). Please follow the dataset providers’ access conditions and cite the original dataset publications.

### Data Format in the Current Repository

The existing [datasets.py](datasets.py) includes loaders for `Pose_300W_LP`, `AFLW2000`, and `BIWI`, as well as additional AFLW and AFW loaders.

- **300W-LP / AFLW2000:** paired `.jpg` images and `.mat` annotations, with a text file listing sample paths relative to the dataset directory **without file extensions**.
- **BIWI:** a preprocessed NumPy `.npz` archive containing `image` and `pose` arrays. The loader reads pose columns in **yaw, pitch, roll** order, in degrees, before converting them to radians. Raw BIWI downloads must be preprocessed into this format.
- [filename.py](filename.py) contains a helper for generating image filename lists from paired image/annotation files.

The required MAT annotation and rotation helpers are included in [utils.py](utils.py). Model, training, evaluation, and raw BIWI preprocessing code are not included in the current public snapshot.

### Using the Data Utilities

Use Python 3.10 or later and install the data utility dependencies:

```bash
python -m pip install -r requirements.txt
python filename.py --root_dir /path/to/300W_LP --filename files.txt
```

A relative output filename is resolved under the dataset root. The helper writes sorted, extension-free relative paths; excludes poses outside ±99°; and reports images with missing MAT annotations. It does not change the working directory.

For deterministic data loading, pass `train_mode=False` to `getDataset`. Training-time horizontal flips update both the rotation matrix and angle labels. A custom transform should preserve pose geometry, unless it also updates the targets.

## Citation

If you find this work useful, please cite:

```bibtex
@article{liu2026semanticplasticity,
  title   = {Learning Semantic Plasticity Relationship via Hierarchical Correlational Modeling for Robust Head Pose Estimation},
  author  = {Liu, Tingting and Chen, Qiang and Li, Xi and Xia, Haoting and Liu, Zhibing and Liu, Hai and Liu, Qiuxia and Li, You-Fu},
  journal = {IEEE Transactions on Circuits and Systems for Video Technology},
  year    = {2026},
  doi     = {10.1109/TCSVT.2026.3747988},
  url     = {https://doi.org/10.1109/TCSVT.2026.3747988}
}
```


## Acknowledgments

We thank the original authors and the providers of 300W-LP, AFLW2000-3D, and BIWI for making their resources available.
