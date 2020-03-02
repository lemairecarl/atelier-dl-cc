# Atelier de Deep Learning sur Calcul Canada

**Recherche d'hyperparamètres**

Ces instructions sont une version abrégée de la
[version sur la branche master](https://github.com/lemairecarl/atelier-dl-cc), avec modifications pour faire une
recherche d'hyperparamètres. L'accent est mis sur une
recherche d'hyperparamètres très simple (de type grid search).

## Concevoir la séquence de commandes

Vous devriez toujours essayer votre séquence de commandes avec une tâche interactive (`salloc`) avant de vous lancer
dans la section suivante (`sbatch`).

## Soumettre la tâche

Créez le fichier `atelier.sh`. Ajoutez-y les lignes suivantes:

```bash
#!/bin/bash
#SBATCH --gres=gpu:k20:1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-5:10:00  # DD-HH:MM:SS

# We will write directly in the project storage. WHEN YOU DO THIS, MAKE SURE YOUR PROGRAMS AND SCRIPTS READ AND WRITE
AT A LOW FREQUENCY. 1 READ OR WRITE PER SECOND IS CONSIDERED HIGH FREQUENCY!
OUTDIR=~/project/out/$SLURM_JOB_ID
mkdir -p $OUTDIR

cd $SLURM_TMPDIR

module load python/3.6 cuda cudnn
virtualenv --no-download $SLURM_TMPDIR/env  # SLURM_TMPDIR is on the compute node
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r ~/atelier-dl-cc/requirements.txt

mkdir $SLURM_TMPDIR/data
tar xf ~/project/tinyimagenet.tar -C $SLURM_TMPDIR/data  # Transfer all data

python ~/atelier-dl-cc/main.py $SLURM_TMPDIR/data --save-path $OUTDIR/run1 --wd 0.0000
python ~/atelier-dl-cc/main.py $SLURM_TMPDIR/data --save-path $OUTDIR/run2 --wd 0.0001
python ~/atelier-dl-cc/main.py $SLURM_TMPDIR/data --save-path $OUTDIR/run3 --wd 1.0000
```

Enregistrez le fichier, et soumettez-le:

    sbatch atelier.sh

## Comparer les choix d'hyperparamètres

Créez un tunnel entre votre ordinateur et le serveur:

    ssh -N -f -L localhost:6006:localhost:6006 <username>@helios3.calculquebec.ca

Sur le noeud de login, allez dans le dossier qui contient les expériences à comparer, et vérifiez que les fichiers
`events.out.tfevents*` sont présents. Si `find` n'affiche rien, c'est mauvais signe.

    cd ~/project/out/1234  # remplacer 1234
    find . -name 'events.out.tfevents*'

Installez TensorBoard sur le noeud de login (si ce n'est déjà fait):

    module load python/3.6
    virtualenv tbenv
    source tbenv/bin/activate
    pip install tensorboard

Lancez TensorBoard en spécifiant le bon dossier:

    source tbenv/bin/activate
    tensorboard --logdir=~/project/out/1234 --host 0.0.0.0 --port 6006
