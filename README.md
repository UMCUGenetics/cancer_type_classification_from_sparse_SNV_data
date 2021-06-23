## cancer_type_classification_from_sparse_SNV_data
Scripts used in the **'Cancer type classification in liquid biopsies based on sparse mutational profiles enabled through data augmentation and integration'** manuscript (https://biorxiv.org/cgi/content/short/2021.03.09.434391v1).

*** code upload is in progress ***

### Data augmentation on sparse data
  *Used PCAWG .vcf files are available upon request from the ICGC data portal*
  
  Sparse SNV samples can be generated with the script in 'sparse_data_generation/'.
  Individual sparse samples then need to be merged into a final input data matrix.
     
### Classification models
  The available classification models are extensions of the baseline model (Jiao, W., Atwal, G., Polak, P. et al. A deep learning system accurately classifies primary and metastatic cancers using passenger mutation patterns. Nat Commun 11, 728 (2020). https://doi.org/10.1038/s41467-019-13825-8) which is available under Apache 2.0 license: https://github.com/ICGC-TCGA-PanCancer/TumorType-WGS/tree/master/DNN-Model  

### Feature importance assessment
  Scripts are based on https://github.com/ankurtaly/Integrated-Gradients and were adjusted for the models used in this work
