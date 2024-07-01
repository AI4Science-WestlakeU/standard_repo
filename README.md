# project_module
This is a project module for a new project, which inludes the file framework, DL pipline, RM.md example and some useful utils.
**To be updating now.**

Any questions, please contact by zhangtao@westlake.edu.cn

# paper_name

[Paper](URL) | [arXiv](URL) | [Poster](URL) | [Tweet](URL) 

Official repo for the paper [Paper Name](URL).<br />
[Author name](https://zhangtao167.github.io)
ICLR 2024 **spotlight**. 

We propose a novel XXX.

Framework of paper:
<a href="url"><img src="/assets/fig1.png" align="center" width="600" ></a>

## Installation


1. Install dependencies.

```code
conda create -n ENV_NAME python=3.x.x
```

Install dependencies:
```code
pip install -r requirements.txt
```

#  file structure
- project_module
  - data                    # data class and dataloader used in the project
  - dataset                 # datasets ready for training or analysis
  - config                  # configuration files for training and inference
  - inference               # scripts for model inference
  - model                   # model definitions
  - results                 # results and logs from training and inference
  - train                   # Scripts and configuration files for training models
  - utils                   # Utility scripts and helper functions
  - .gitignore              # Specifies intentionally untracked files to ignore by git
  - filepath.py             # Python script for file path handling
  - README.md               # Markdown file with information about the project for users
  - reproducibility_statement.md # Markdown file with statements on reproducibility practices
  - requirements.txt        # Text file with a list of dependencies to install


## Dataset and checkpoint

All the dataset can be downloaded in this [this link](URL). Checkpoints are in [this link](URl). Both dataset.zip and checkpoint_path.zip should be decompressed to the root directory of this project.


## Training

Below we provide example commands for training the diffusion model/forward model.

### training model


```code
python train_1d.py 
```


## inference

Here we provide commands for inference using the trained model:

### model 1
```code
python inference.py
```


## Related Projects
  
* [NAME](URL) (ICLR 2023 spotlight): brief description of the project.

Numerous practices and standards were adopted from [CinDM](https://github.com/AI4Science-WestlakeU/cindm).
## Citation
If you find our work and/or our code useful, please cite us via:

```bibtex
@inproceedings{
    ...
}
```
