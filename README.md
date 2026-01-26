# ESPNet: Edge-Aware Feature Shrinkage Pyramid for Polyp Segmentation (MICCAI'25)

### Architecture
![ESPNet Architecture](figs/architecture.png)

## Training Dataset
- You can download the training data we compiled from here: https://drive.google.com/file/d/16zqdG_3PJaRRkm2wahpE2mp-gyn8rFUF/view?usp=sharing
- You can also download the pretrained transformer here: https://drive.google.com/file/d/1OmE2vEegPPTB1JZpj2SPA6BQnXqiuD1U/view?usp=share_link
- You will need to add their paths to train.sh in scripts.
- When creating edge maps, we recommend dilating them; it produces better results.

## Evaluation
- You need to add the test datasets' paths in test.py, and the model checkpoint and results paths in test.sh (or modify test.py).
- You can download the checkpoint from here: https://drive.google.com/file/d/161dyhkMQtXeF4EbykIGGMbXLBQ16Qbfp/view?usp=drive_link

## Prerequisites

- Creating a virtual environment in terminal: `conda create -n ESPNet python=3.8`.
- Versions: torch==2.4.1 scipy==1.10.1 torchvision==0.7.0 timm==0.5.4
  

## Results

### Seen Datasets
![Seen Dataset Results](figs/results_seen.png)

### Unseen Datasets
![Unseen Dataset Results](figs/results_unseen.png)


## 📄 Citation

If you use this code or find our work helpful, please cite our paper:

> Raneem Toman, Venkataraman Subramanian, and Sharib Ali.  
> "**ESPNet: Edge-Aware Feature Shrinkage Pyramid for Polyp Segmentation**".  
> *Proceedings of the International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 2025.

### BibTeX
```bibtex
@inproceedings{toman2025espnet,
  author    = {Raneem Toman and Venkataraman Subramanian and Sharib Ali},
  title     = {ESPNet: Edge-Aware Feature Shrinkage Pyramid for Polyp Segmentation},
  booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year      = {2025},
}
