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