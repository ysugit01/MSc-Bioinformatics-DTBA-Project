## MSc Bioinformatics Project
==================

#__Comparison of Methods for the Prediction of Binding Affinity__

__Author:__             Yuri Benstead
__Author Email:__       ysugit01@mail.bbk.ac.uk
__Supervisor:__         Mark Williams
__Supervisor Email:__   ma.williams@bbk.ac.uk
__Deadline:__           25-August-2023


##Overview
========

The ability to use structural information to understand and predict the binding affinity of small molecules is an important, and thus far only partly achieved, goal of stucture-based drug design for which many methods have been developed. This project aims to implement and test several current binding affinity prediction methods to try to identify the best method.

The first part of the project would involve building a test set of protein-ligand complexes for which both structural and affinity data are known. Toward this end, BindingDB has released a collection of 1200 sets of experimental data with known related structures https://www.bindingdb.org/rwd/validation_sets/index.jsp. These sets would seem to be potentially very useful, however, they also seem to have several problems. It is not clear which structures and data are ‘exact’ matches (i.e. the same small molecule and protein construct), which would be needed for accurate testing, or merely close matches (e.g. same protein but only a similar small molecule). The sets contain a lot of redundant data or data collected under different temperatures or solution conditions, which would affect the affinity. Consequently, it will be necessary to develop a programmatic pipeline to examine and clean up these datasets.

The second part of the project would be to implement several methods for affinity prediction and automate their use so that hundereds or thousands of protein-small molecule systems could be evaluated. The results of prediction would be finally compared to experiment.


##BindingDB Source Data
=======================

Due to limitations on the amount of data that can be stored within Github, the Alteryx database created from the BindingDB website has not been included however a copy of the workflows used to generate the various drug-target matrixes used for this project + SMILEs data used to generate the drug-drug data files have been included in the 'Alteryx' folder. Please note that the free trial of the software has now expired so a demo will only be possible if another free trial can be obtained.


##Software Dependencies
=====================

All of the following python libraries have been installed into the 'thoth' environment of the university servers for the purpose of this project:

enum            [v.xxx]
igraph          [v.xxx]
lifelines       [v.xxx]
numpy           [v.xxx]
matplotlib      [v.xxx]
os              [v.xxx]
pandas          [v.xxx]
rdkit           [v.xxx]
rlscore         [v.0.8]
scipy           [v.xxx]
seaborn         [v.xxx]
sklearn         [v.xxx]
sys             [v.xxx]
xgboost         [v.xxx]

A C-compiler was also installed for RLScore (to build Cython extensions)


##Software Modules
==================
List of software modules written for this project (available in the Project folder):
*Generate Drug Similarity File.py
*Generate Target Similarity File.py
*KronRLS (Davis) - Gaussian Kernel.py
*KronRLS (Davis).py
*KronRLS (GPCR - 595 x 31) - Gaussian Kernel.py
*KronRLS (GPCR - 595 x 31).py
*KronRLS (GPCR - 746 x 12) - Gaussian Kernel.py
*KronRLS (GPCR - 746 x 12).py
*KronRLS (GPCR - 878 x 56) - Gaussian Kernel.py
*KronRLS (GPCR - 878 x 56).py
*SimBoost (Davis).py
*SimBoost (GPCR - 595 x 31).py
*SimBoost (GPCR - 746 x 12).py
*SimBoost (GPCR - 878 x 56).py
*TwoStepRLS (Davis) - Gaussian.py
*TwoStepRLS (Davis).py
*TwoStepRLS (GPCR - 595 x 31) - Gaussian.py
*TwoStepRLS (GPCR - 595 x 31).py
*TwoStepRLS (GPCR - 746 x 12) - Gaussian.py
*TwoStepRLS (GPCR - 746 x 12).py
*TwoStepRLS (GPCR - 878 x 56) - Gaussian.py
*TwoStepRLS (GPCR - 878 x 56).py


##How to Run The Code
================

'ssh -X thoth' (enter password)
'module load python/v3'
'conda activate rlscore'
Navigate to main 'Project' folder
'python3 <name of module>'



##Citations
==============

*RLScore library Github page: https://github.com/aatapa/RLScore/tree/master
*mahtaz's SimBoost Github page: https://github.com/mahtaz/Simboost-ML_project-


##Credits
=======

- Mark Williams: Thank you for all your help, support and understanding to deliver this complex project whilst I was very ill with Covid and post-Covid over 6 months from the beginning of 2023.
- Dr David Houldershaw: Special thanks for helping me with installations of various libraries on the university sever when my PC broke down a week before the deadline.
