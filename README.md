# DetectVul
This repository is still undergoing refactoring


## Getting started

### Install required packages
```
sh install.sh
```

### Run server

```
python be/app.py
```

### Run Client
```
cd fe
yarn dev
```

### Demo
Folder `examples` contain Python source files that contain vulnerabilities which are collected from GitHub issues in other projects.
![image info](sample.png)

 You can copy and paste source code of these example files into the input box and tap the detect button, the predicted vulnerable statements will be marked by red lines in the output box.


### Bibtex

Original Paper: https://www.sciencedirect.com/science/article/abs/pii/S0167739X24004680

bibtex

```
@article{TRAN2024107504,
 title = {DetectVul: A statement-level code vulnerability detection for Python},
 journal = {Future Generation Computer Systems},
 pages = {107504},
 year = {2024},
 issn = {0167-739X},
 doi = {https://doi.org/10.1016/j.future.2024.107504},
 url = {https://www.sciencedirect.com/science/article/pii/S0167739X24004680},
 author = {Hoai-Chau Tran and Anh-Duy Tran and Kim-Hung Le},
 keywords = {Source code vulnerability detection, Deep learning, Natural language processing},
 abstract = {Detecting vulnerabilities in source code using graph neural networks (GNN) has gained significant attention in recent years. However, the detection performance of these approaches relies highly on the graph structure, and 
 constructing meaningful graphs is expensive. Moreover, they often operate at a coarse level of granularity (such as function-level), which limits their applicability to other scripting languages like Python and their effectiveness in identifying 
 vulnerabilities. To address these limitations, we propose DetectVul, a new approach that accurately detects vulnerable patterns in Python source code at the statement level. DetectVul applies self-attention to directly learn patterns and 
 interactions between statements in a raw Python function; thus, it eliminates the complicated graph extraction process without sacrificing model performance. In addition, the information about each type of statement is also leveraged to enhance 
 the model’s detection accuracy. In our experiments, we used two datasets, CVEFixes and Vudenc, with 211,317 Python statements in 21,571 functions from real-world projects on GitHub, covering seven vulnerability types. Our experiments show that 
 DetectVul outperforms GNN-based models using control flow graphs, achieving the best F1 score of 74.47%, which is 25.45% and 18.05% higher than the best GCN and GAT models, respectively.}
}
```
