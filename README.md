# Atelier de Deep Learning sur Calcul Canada

Cet atelier est basé sur les bonnes pratiques pour l'apprentissage machine sur les grappes de Calcul Canada. Ces bonnes
pratiques sont élaborées en détail dans les références suivantes:

* [Tutoriel Apprentissage Machine](https://docs.computecanada.ca/wiki/Tutoriel_Apprentissage_machine)
* [Guide Apprentissage Machine](https://docs.computecanada.ca/wiki/AI_and_Machine_Learning/fr)
* [Diaporama "Using Compute Canada Clusters for Machine Learning Research"](https://docs.google.com/presentation/d/1B978yexo6nBLAVusICLCs-QKwrK1v49T3E4GjNfFLgs/edit?usp=sharing)

La présente version de l'atelier est conçue pour le cours IFT725 donné à l'Université de Sherbrooke à l'hiver 2020.

## Se connecter au serveur

Nous nous connecterons à [la grappe Hélios](https://docs.computecanada.ca/wiki/H%C3%A9lios), située à l'Université Laval, à Québec.

Ouvrez un terminal, et lancez la commande suivante:

       ssh <username>@helios3.calculquebec.ca

Dans cet environnement, vous pouvez:
* Préparer vos données
* Préparer votre code
* Préparer votre script de soumission
* Et toute autre opération, qui n'est pas l'entraînement.
      
Pour l'instant vous n'avez ni données, ni code. Nous allons régler ça dans la prochaine section.

## Transférer des données et du code

1. Téléchargez la BDD TinyImageNet à ce lien: https://drive.google.com/file/d/1g_MSfNMySQyhgqL8OIoP-nk3ogJCgWRM/view?usp=sharing
2. Transférez le fichier sur Hélios:

       rsync tinyimagenet.tar <username>@helios3.calculquebec.ca:
       
   Le `:` à la fin de la ligne est important.

3. Déplacez le fichier dans votre espace de stockage "scratch"<sup>[1](#footnote1)</sup>.
   
       ssh <username>@helios3.calculquebec.ca
       mv tinyimagenet.tar ~/scratch

4. Clonez le code dans votre `home` (`home/<username>`, c'est le dossier dans lequel vous arrivez lors d'une connexion
   `ssh`):

       git clone https://github.com/lemairecarl/atelier-dl-cc.git

## Concevoir la séquence de commandes

À cette étape, il s'agit de trouver la bonne séquence de commandes qui permet d'effectuer correctement l'entraînement
sans supervision. Une fois cette séquence trouvée, on en fera un script (section suivante).

1. Soumission d'une tâche interactive. Demandez 2 CPUs, 8GiB de RAM et un GPU, pour une heure:

       salloc --cpus-per-task=2 --mem=8G --gres=gpu:1 --time=1:00:00

   Vous êtes maintenant sur un noeud de calcul, dans une tâche interactive.

2. Changez de dossier pour aller sur le stockage local au noeud de calcul:

       cd $SLURM_TMPDIR

3. Chargez les [modules](https://docs.computecanada.ca/wiki/Utiliser_des_modules) dont nous aurons besoin:

       module load python/3.8
       
4. Créez l'environnement virtuel python, et activez-le:

   **TODO: où placer l'env?? requirements??**
       
       virtualenv --no-download env
       source env/bin/activate
       
5. Installez les paquets python nécessaires:

       pip install --no-index -r ~/atelier-dl-cc/requirements.txt

6. Transférez les données sur le noeud de calcul. Il faut transférer de "project" vers le noeud de calcul
   un seul fichier, dans ce cas-ci, un `.tar`. Le fichier sera extrait sur le noeud de calcul, et non dans le stockage
   partagé.
   
       mkdir data  # nous sommes toujours dans $SLURM_TMPDIR
       cd data
       cp ~/scratch/tinyimagenet.tar .
       tar xf tinyimagenet.tar

7. Vous pouvez maintenant lancer l'entraînement:

       cd $SLURM_TMPDIR
       python ~/atelier-dl-cc/main.py ./data
   
   Si vous voyez des barres de progression apparaître, bravo! L'entraînement est lancé avec succès. Vous pouvez le stopper.
   
8. Effectuez les corrections nécessaires, s'il y a lieu. Notez les commandes, car elles iront dans le script à la
   section suivante.

## Soumettre une tâche

**TODO CETTE SECTION**

Créez le fichier `atelier.sh`. Vous pouvez créer le fichier sur votre laptop pour le transférer ensuite, ou vous pouvez
le créer directement sur le serveur, en utilisant `nano` ou `vim`. Ajoutez-y les lignes suivantes:

```
#!/bin/bash
#SBATCH --gres=gpu:k20:1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-2:00:00  # DD-HH:MM:SS
```
Ces lignes vont remplacer les arguments à la commande `salloc` utilisée ci-haut.

Ensuite, ajoutez la séquence de commandes que vous avez validée dans la section prédécente.

Pour terminer, ajoutez les lignes suivantes. Elles servent à conserver les résultats de l'entraînement, qui autrement seraient
effacées lors de la fin de la tâche.

    OUTDIR=~/project/out/$SLURM_JOB_ID
    mkdir -p $OUTDIR
    cp -r lightning_logs/version*/* $OUTDIR

Enregistrez le fichier, et soumettez-le:

    sbatch atelier.sh

Vous pouvez vérifier vos tâches actives avec la commande `sq`. Les tâches en attente ont le statut `PD` (pending), et
les tâches en cours, `R` (running).

## Récupérer les sorties de la tâche

Dans le répertoire où vous avez appelé `sbatch`, un nouveau fichier a été créé, au nom similaire à `slurm-XXXX.out`.
Ce fichier contiendra la sortie standard (les _print_) et les erreurs déclenchées par le script de soumission et le
le script d'entraînement (`main.py`).

Pour voir la sortie, utilisez le programme `less`, qui vous permet d'afficher dans le terminal un fichier page par page:

    less slurm-XXXX.out
    
Utiliser "Page Up" et "Page Down" pour naviguer, et "q" pour quitter.

## Suivre le déroulement

Cette section est facultative, mais recommandée.

### Se connecter au noeud de calcul

Ouvrez un nouveau shell sur la grappe:

    ssh <username>@phoenix.calculquebec.cloud
    sq
    
La liste de vos tâches en cours (ou en attente) s'affiche. Notez l'identificateur du noeud (colonne NODELIST), de la forme `nodeX`. Ensuite utilisez cet identificateur pour vous connecter au noeud:
    
    ssh <id_noeud>

### Vérifier que le GPU est utilisé

Une fois connecté au noeud de calcul, exécutez:

    watch nvidia-smi
    
Vérifiez que % d'utilisation ne reste pas à zéro.

### Suivre les métriques avec _Tensorboard_

Sur votre ordinateur local, exécutez (remplacez les variables):

    ssh -N -f -L localhost:6006:<id_noeud>:6006 <username>@phoenix.calculquebec.cloud

Vous devrez peut-être changer le port 6006 pour un autre si vous avez l'erreur `Address already in use`.

Ensuite, connectez-vous au noeud de calcul, et rendez-vous dans le dossier suivant (remplacez les variables):

    cd /localscratch/<username>.<job_id>.0
    
Ensuite:
    
    source env/bin/activate
    tensorboard --logdir=lightning_logs --host 0.0.0.0 --port 6006

Remplacez le port 6006 selon ce que vous avez utilisé ci-haut. (Même chose pour l'étape suivante.)

Finalement, ouvrez votre navigateur internet à l'adresse `localhost:6006`.

## Recherche d'hyperparamètres

**TODO**

Reférez-vous au README sur la branche [`hpsearch`](https://github.com/lemairecarl/atelier-dl-cc/tree/hpsearch).

## Réponses aux questions fréquentes

* Dans `atelier.sh`, ne mettez pas de ligne `salloc`.
* Lancez la commande `sbatch` sur un noeud de login, et non sur un noeud de calcul (obtenu avec un `salloc`).
* Vous pouvez lister vos tâches actives avec la commande `sq`, et annuler une tâche avec la commande `scancel <job_id>`.
    * "R" veut dire que la tâche est en exécution (running)
    * "PD" veut dire que la tâche est en attente (pending)
* Si vous obtenez l'erreur `Address already in use`, essayez de changer le port. Au lieu de 6006, essayez 6007, 6008,
  etc. Ensuite, changez `--port 6006` dans la commande `tensorboard`.
* La commande `ssh -N -f -L localhost ...` quitte tout de suite si elle fonctionne.
* Si en ouvrant Tensorboard, vous avez l'avertissement `Tensorflow not found`, vous pouvez l'ignorer.
* Si votre script `atelier.sh` contient moins de 10 commandes, relisez les instructions.

---

### Notes

1. <a name="footnote1"></a>Dans le cadre de cet atelier, nous avons seulement accès à "scratch". Dans la vraie vie, vous aurez aussi "project"; ce dernier vous permettra de stocker des données à long terme, alors que "scratch" est purgé périodiquement. Pour plus de détails,
référez-vous à notre [documentation sur l'espace project](https://docs.computecanada.ca/wiki/Project_layout/fr).
