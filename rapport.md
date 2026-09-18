### question 1.C

GPU  Name  : Persistence-M 

### question  1.D

scancel 1506

### question 1.E

fichier :hello-slurm-1530.out   --> On voit le message :"Bonjour depuis SLURM" et on voit des infos sur les GPU .
et y a aussi : hello-slurm-1530.err  


### question 1.F

commandline : sacct -j 1506 --format=JobID,State,Elapsed,MaxRSS,ReqMem,ReqCPUS

ReqMem correspond à la mémoire qu'on a demandé à SLURM lorsqu'on a soumis le job.
MaxRSS correspond à la mémoire RAM réellement utilisée au maximum pendant l'exécution. 

## Exo2

to do pour activer mamba : `source ~/miniforge3/etc/profile.d/conda.sh`

### question 2.B 
Pour vérifier la version du Python : `python --version`
Python 3.10.21

Pour le binaire : `which python`
/mnt/hdd/homes/daberkane/miniforge3/envs/deeplearning/bin/python

### question 2d

PyTorch version: 2.14.0+cu130
CUDA available: True
Device count: 1
Device 0 name: NVIDIA L4

## Exo3 

### question 3a

mettre image


### question 3b
X  : (N, 3)
W1 : (m, 3)
b1 : (1, m) -> diffusé en (N, m)
H  : (N, m)
W2 : (t, m)
b2 : (1, t) -> diffusé en (N, t)
Y  : (N, t)

### question 3c

TODO : mettre image

### question 3d

TODO : mettre image 


### question 3e

- Pourquoi utilisons-nous la règle de la chaîne (chain rule) pour calculer les gradients dans les réseaux de neurones profonds ?

TODO reprendre ce qui est écrit dans le cahier 

- Quelles sont les principales raisons d'utiliser des mini-batchs plutôt que d'optimiser sur un seul exemple à la fois ou sur l’ensemble total des données ?

TODO TODO reprendre ce qui est écrit dans le cahier


### question 3f

TODO : mettre image 