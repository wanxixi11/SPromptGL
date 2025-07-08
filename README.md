# 📄 SPromptGL: Semantic Prompt Guided Graph Learning for Multi-modal Brain Disease

> **SPromptGL: Semantic Prompt Guided Graph Learning for Multi-modal Brain Disease**  
> **Xixi Wan**, Bo Jiang\*, Shihao Li, and Aihua Zheng\* 
> Accepted at **[MICCAI, 2025, Accepted]**

[📄 Paper Link]()

---

## 🧠 Overview

This repository contains the official implementation of our paper:

> "**SPromptGL: Semantic Prompt Guided Graph Learning for Multi-modal Brain Disease**"  
> Xixi Wan, Bo Jiang\*, Shihao Li, and Aihua Zheng\*   
> Medical Image Computing and Computer Assisted Intervention 2025


We propose **Semantic Prompt-guided Graph Learning (SPromptGL)**, a novel approach for multi-modal disease prediction that captures the discriminative regions of different modalities while enhancing their interaction and fusion. 

---

## Step 1: Data preprocessing
Running the code of data preprocessing in ./data/{dataset}/xxx.ipynb to preprocess the raw data to standard data as the input of SPromptGL.

For more details, please refer to [MMGL](https://github.com/SsGood/MMGL).

## Step 2: Training and test

Running 
```
sh ./{dataset}-simple-2-concat-weighted-cosine.sh
```

Notice: the sh file is used to reproduce the result reported in our paper, you could also run this script to train and test your own dataset:
```
python main.py
```
Besides, you can modify 'network.py' to establish a variant of SPromptGL and 'model.py' to try and explore a better training strategy for other tasks.



## ✏️ Citation


```BibTeX
@inproceedings{wan2025SPromptGL,
  title={SPromptGL: Semantic Prompt Guided Graph Learning for Multi-modal Brain Disease},
  author={Xixi Wan, Bo Jiang, Shihao Li, and Aihua Zheng},
  booktitle={Medical Image Computing and Computer Assisted Intervention},
  year={2025}
}
```
