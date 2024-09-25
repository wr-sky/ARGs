# ARGs
Predicting Antibiotic Resistance Genes and Bacterial Phenotypes Based on Protein Language Models

The codes and data included are the supplementary sources of paper "Predicting Antibiotic Resistance Genes and Bacterial Phenotypes Based on Protein Language Models"

Data: The ARGs and strains utilized, which contains 5 files as below.
Data/same_HDM_deepARG.fasta: The same ARGs utilized by DeepARG and HMD-ARG.
Data/difference_deepARG.fasta: The unique ARGs in deepARG, which are different from these in HMD-ARG.
Data/difference_HDM.fasta: The unique ARGs in HMD-ARG, which are different from these in deepARG.
Data/HyperVR.fasta: 2,000 non-resistant genes reported in HyperVR.
Data/AST_NCBI_id.txt: The trains utilized in antibiotic susceptibility prediction task, including the NCBI id and phenotypes of each strain.

Code: The codes utilized to predict ARGs of target genes and antibiotic susceptibility of target bacterial strains, which contains 24 files categorized as below.
Code/1_Extraction+Enhancement: Codes utilized for feature extraction and data enhuancement.
Code/2_Classfication+Integration: Codes utilized for classification and results integration.
Code/3_CARD: Codes utlized for RGI selection in transfer application of the antibiotic susceptibility prediction task.
