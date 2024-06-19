# Protein language models and machine learning facilitate the identification of antimicrobial peptides.

This repository contains the source code and relevant information for the implementations and use cases presented in the work: <br>
David Medina-Ortiz<sup>1,2*</sup>, Seba Contreras<sup>3*</sup>, Diego Fernández<sup>1</sup>, Nicole Soto-García<sup>1</sup>, Iván Moya<sup>1,4</sup>, and Álvaro Olivera-Nappa<sup>2</sup>.<br>

Protein language models and machine learning facilitate the identification of antimicrobial peptides. <br>

https://doi.org/XXXX<br>

<sup>*1*</sup><sub>Departamento de Ingeniería en Computación, Universidad de Magallanes, Av. Pdte. Manuel Bulnes 01855, 6210427, Punta Arenas, Chile.</sub> <br>
<sup>*2*</sup><sub>Centre for Biotechnology and Bioengineering, CeBiB, Universidad de Chile, Avenida Beauchef 851, 8320000, Santiago, Chile.</sub> <br>
<sup>*3*</sup><sub>Max Planck Institute for Dynamics and Self-Organization, Am Fa\ss berg 17, 37077 Göttingen, Germany.</sub> <br>
<sup>*4*</sup><sub>Departamento de Ingeniería Química, Universidad de Magallanes, Av. Pdte. Manuel Bulnes 01855, 6210427, Punta Arenas, Chile.</sub> <br>
<sup>*\**</sup><sub>Corresponding author</sub> <br>

---
## Table of Contents
- [A summary of the proposed work](#summary)
- [Requirements and instalation](#requirements)
- [Raw data](#data)
- [Implemented pipeline](#data)
- [Demos](#data)
- [Results and models](#data)
---

<a name="summary">Requirements and instalation</a>

# Protein language models and machine learning facilitate the identification of antimicrobial peptides.

Peptides are bioactive molecules whose functional versatility in living organisms has successfully found applications in diverse industrial fields. 
In recent years, the amount of data describing peptide sequences and function collected in open repositories has substantially increased, allowing the application of more complex computational models to study the relations between peptide composition and function.
This work introduces sequence-based classification models for detecting peptides' functional biological activities, focusing on accelerating the discovery and *de novo* design of potential antimicrobial peptides (AMPs). 
A novel sequence-based pipeline was developed to train binary classification models, integrating protein language models and machine learning algorithms. This pipeline produced 21 models targeting antimicrobial, antiviral, and antibacterial activities, achieving an average precision exceeding 83%. Benchmark analyses revealed that our models outperformed existing methods for AMPs and delivered comparable results for other biological activities. Utilizing the Peptide Atlas, we discovered over 300,000 potential AMPs and demonstrated an integrative approach with generative learning to aid in the *de novo* design, resulting in over 500 novel AMPs. 
The combination of our methodology, robust models, and generative design strategy highlights a significant advancement in peptide-based drug discovery and represents a pivotal tool for therapeutic applications.

<a name="requirements">Requirements and Install process</a>

The requirements are summarized in the [environment.yml](environment.yml) file. Some requirements are summarized below:

- Python version 3.9+
- bio-embeddings
- scikit-learn
- xgboost

Once this repository is cloned, please run the following command:

```
    conda env create -f environment.yml
```
<a name="data">Raw data and collection process</a>