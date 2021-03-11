## cancer_type_classification_from_sparse_SNV_data
Scripts used in the **'Cancer type classification in liquid biopsies based on sparse mutational profiles enabled through data augmentation and integration'** manuscript.


### Data augmentation on sparse data
  *Used PCAWG .vcf files are available upon request from the ICGC data portal*
  
  Sparse SNV samples can be generated with the script in 'sparse_data_generation/'.
  Individual sparse samples then need to be merged into a final input data matrix (see the 10% data example matrices in 'data/').
     
### Classification models
  The available classification models are extensions of the baseline model which is available here under Apache 2.0 license: https://github.com/ICGC-TCGA-PanCancer/TumorType-WGS/tree/master/DNN-Model 

### Feature importance assessment
  Scripts are based on https://github.com/ankurtaly/Integrated-Gradients and were adjusted for the models used in this work
