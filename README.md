# Atelier de Deep Learning sur Calcul Canada

Cet atelier est basé sur les bonnes pratiques pour l'apprentissage machine sur les grappes de Calcul Canada. Ces bonnes
pratiques sont élaborées en détail dans les références suivantes:

* [Tutoriel Apprentissage Machine](https://docs.computecanada.ca/wiki/Tutoriel_Apprentissage_machine)
* [Guide Apprentissage Machine](https://docs.computecanada.ca/wiki/AI_and_Machine_Learning/fr)
* [Diaporama "Using Compute Canada Clusters for Machine Learning Research"](https://docs.google.com/presentation/d/1B978yexo6nBLAVusICLCs-QKwrK1v49T3E4GjNfFLgs/edit?usp=sharing)

La présente version de l'atelier est conçue pour le cours IFT780 donné à l'Université de Sherbrooke à l'hiver 2021.

## 1. Préparation

### Se connecter au serveur

[Calcul Canada met plusieurs grappes à la disposition des chercheurs canadiens.](https://www.computecanada.ca/page-daccueil-du-portail-de-recherche/acces-aux-ressources/ressources-disponibles/?lang=fr) Pour cet atelier, nous nous utiliserons une grappe virtuelle, qui sera détruite peu après l'atelier. Cette grappe virtuelle offre une expérience pratiquement identique à celle des grappes réelles.

* **Sous Windows:**
    * [Téléchargez MobaXterm ici](https://mobaxterm.mobatek.net/)
    * Pour vous connecter à un serveur auquel vous ne vous êtes pas connecté auparavant : sous Sessions->New session, sélectionnez SSH puis entrez l'adresse du serveur (`ift780.calculquebec.cloud`) et votre nom d'utilisateur (s'il y a lieu, cochez Specify username). Cliquez sur OK. MobaXTerm enregistre ces renseignements pour les connexions ultérieures au serveur et établit la connexion SSH.

* **Sous Linux/MacOS:** Ouvrez un terminal, et lancez la commande suivante:

       ssh <username>@ift780.calculquebec.cloud

Une fois connecté au serveur, vous pouvez:
* Préparer vos données
* Préparer votre code
* Préparer votre script de soumission
* Et toute autre opération, qui n'est pas l'entraînement.
      
Pour l'instant vous n'avez ni données, ni code. Nous allons régler ça dans la prochaine section.

### Transférer des données et du code

1. Téléchargez la BDD TinyImageNet à ce lien: https://drive.google.com/file/d/1g_MSfNMySQyhgqL8OIoP-nk3ogJCgWRM/view?usp=sharing
2. Transférez le fichier sur Hélios:
      * **Sous Windows avec MobaXterm:** Vous devriez voir un onglet "sftp" à gauche du terminal. Les fichiers sur le serveur s'affichent à cet endroit. Pour le fichier, faites simplement un glisser-déposer (drag and drop) dans cette zone.
      * **Sous Linux/MacOS:** Dans un terminal **sur votre ordinateur (pas sur le serveur)**, exécutez ce qui suit:
       
       scp tinyimagenet.tar <username>@ift780.calculquebec.cloud:
       
   Note: Le `:` à la fin de la ligne est important.
   
3. Assurez-vous que le fichier a bien été transféré en faisant un `ls` sur le serveur.

3. Déplacez le fichier dans votre espace de stockage "project"<sup>[1](#footnote1)</sup>.
   
       ssh <username>@ift780.calculquebec.cloud
       mv tinyimagenet.tar ~/projects/def-sponsor00/$USER
       
   Note: `def-sponsor00` correspond au nom du compte de votre superviseur. Pour l'atelier on utilise un compte bidon qui s'appelle `def-sponsor00`. Pour plus de détails, référez-vous à notre [documentation sur l'espace project](https://docs.computecanada.ca/wiki/Project_layout/fr).

4. Clonez le code dans votre `home` (`home/<username>`, c'est le dossier dans lequel vous arrivez lors d'une connexion
   `ssh`):

       git clone https://github.com/lemairecarl/atelier-dl-cc.git

**NOTE** pour les utilisateurs de MobaXterm: Pour coller du texte dans le terminal, vous pouvez faire clic-droit. Pour copier du texte, suffit de le sélectionner avec la souris. Tout texte sélectionné dans le terminal sera automatiquement copié.

## 2. Essayer avec une tâche interactive

À cette étape, il s'agit de trouver la bonne séquence de commandes qui permet d'effectuer correctement l'entraînement
sans supervision. Une fois cette séquence trouvée, on en fera un script (section suivante).

1. Soumission d'une tâche interactive. Demandez 4 CPUs et 3GB de RAM, pour dix minutes:

       salloc --cpus-per-task=4 --mem=3G --time=0:10:00

   Vous êtes maintenant sur un noeud de calcul, dans une tâche interactive.
   
   Notes:
   
   * Vous serez déconnecté du noeud de calcul après 10 minutes.
   * Remarquez que le prompt a changé, avant c'était `username@login1`, maintenant c'est `username@nodeX` où `X` varie.
   

2. Chargez les [modules](https://docs.computecanada.ca/wiki/Utiliser_des_modules) dont nous aurons besoin:

       module load python/3.8
       
3. Changez de dossier pour aller sur le stockage local au noeud de calcul:

       cd $SLURM_TMPDIR
       
4. Créez l'environnement virtuel python, et activez-le:

       virtualenv --no-download env
       source env/bin/activate
       
   Note: c'est aussi possible de créer l'environnement dans votre home (`~`), même si c'est moins recommandé. L'avantage de le mettre dans le home est que vous pouvez créer l'environnement à partir d'un noeud de connexion; ce qui vous donne accès à internet, et donc à plus de paquets python.
       
5. Installez les paquets python nécessaires:

       pip install --no-index -r ~/atelier-dl-cc/requirements.txt

6. Transférez les données sur le noeud de calcul. Il faut transférer de "project" vers le noeud de calcul
   un seul fichier, dans ce cas-ci, une archive `tar`. Le fichier sera extrait sur le noeud de calcul, et non dans le stockage
   partagé.
   
       mkdir data  # nous sommes toujours dans $SLURM_TMPDIR
       cd data
       cp ~/projects/def-sponsor00/$USER/tinyimagenet.tar .
       tar xf tinyimagenet.tar

7. Vous pouvez maintenant lancer l'entraînement (sans utiliser de GPU):

       cd $SLURM_TMPDIR
       python ~/atelier-dl-cc/train.py ./data --gpus 0
   
   Si vous voyez des barres de progression apparaître, bravo! L'entraînement est lancé avec succès. Notez le temps estimé pour l'epoch, qui devrait être autour de 13 minutes.
   
   Ensuite, vous pouvez stopper l'entraînement, avec `Ctrl+C`.
   
8. Pour quitter la tâche interactive, utilisez la commande `exit`. Remarquez que votre prompt redevient `username@login1`.
   
8. Notez les commandes que vous avez utilisés ici, car elles iront dans le script à la
   section suivante.

## 3. Soumettre une tâche

Créez le fichier `train.sh`. Vous pouvez créer le fichier sur votre laptop pour le transférer ensuite, ou vous pouvez
le créer directement sur le serveur, en utilisant `nano` ou `vim`. Ajoutez-y les lignes suivantes:

```bash
#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=3G
#SBATCH --gres=gpu:1
#SBATCH --time=00-00:10:00  # DD-HH:MM:SS
```
Ces lignes vont remplacer les arguments à la commande `salloc` utilisée ci-haut. Cette fois-ci, nous utilisons un GPU.

Pour le paramètre `--time`, assignez une durée qui permettra d'entraîner pendant 4 epochs. Pour estimer une bonne valeur, on peut utiliser la durée d'une epoch sur CPU notée à l'étape précédente; mais divisez cette durée par 8. Un petit calcul vous permettra d'avoir un estimé raisonnable du temps requis sur GPU pour 4 epochs.

Ensuite, ajoutez la séquence de commandes que vous avez validée dans la section prédécente. Ça devrait ressembler à ceci:

```bash
module load python/3.8

# Environnement python
cd $SLURM_TMPDIR
virtualenv --no-download env
source env/bin/activate
pip install --no-index -r ~/atelier-dl-cc/requirements.txt

# Données
mkdir data  # nous sommes toujours dans $SLURM_TMPDIR
cd data
cp ~/projects/def-sponsor00/$USER/tinyimagenet.tar .
tar xf tinyimagenet.tar
cd ..

OUTDIR=~/projects/def-sponsor00/$USER/out/$SLURM_JOB_ID
python ~/atelier-dl-cc/train.py ./data --save-path $OUTDIR --epochs 10
```

**REMARQUE:** Les dernières lignes sont un peu différentes, veuillez utiliser cette nouvelle version.

Enregistrez le fichier, et soumettez-le. **Note: sbatch doit toujours être appelé à partir d'un noeud de connexion (login).**

    sbatch train.sh

Vous pouvez vérifier vos tâches actives avec la commande `sq`. Les tâches en attente ont le statut `PD` (pending), et
les tâches en cours, `R` (running).

### Récupérer les sorties de la tâche

Dans le répertoire où vous avez appelé `sbatch`, un nouveau fichier a été créé, au nom similaire à `slurm-XXXX.out`.
Ce fichier contiendra tout ce qui se serait affiché dans le terminal si vous auriez exécuté la tâche en mode interactif.

Pour voir la sortie, utilisez le programme `less`, qui vous permet d'afficher dans le terminal un fichier page par page:

    less slurm-XXXX.out
    
Utiliser "Page Up" et "Page Down" pour naviguer, et "q" pour quitter.

Pour afficher les 10 dernières lignes du fichier, utilisez ceci:

    tail slurm-XXXX.out

### Vérifier que le GPU est utilisé

Vérifiez d'abord quel est le _job ID_ de votre tâche:

    sq

Ensuite, exécutez:

    srun --jobid=<JOBID> --pty watch nvidia-smi
    
Vérifiez que % d'utilisation (`GPU-Util`) ne reste pas à zéro. Faites `Ctrl+C` pour quitter.

### Suivre les métriques avec _Tensorboard_

1. Créez un environnement python dans votre home. Vous y installerez TensorBoard:

       module load python/3.8
       virtualenv env
       source env/bin/activate
       pip install tensorboard

2. Lancez TensorBoard

       source env/bin/activate  # si pas déja fait
       tensorboard --load_fast false --logdir ~/projects/def-sponsor00/$USER/out --host 0.0.0.0 --port 0
       
   >**Note:** Ici on lance TensorBoard sur un noeud de connexion. C'est plus simple à expliquer dans un atelier, mais ce n'est pas recommandé par Calcul Canada (CC). CC demande à ses utilisateurs de lancer TensorBoard sur les noeuds de calcul. Veuillez vous référer à la [documentation de CC à ce sujet](https://docs.computecanada.ca/wiki/TensorFlow/fr#TensorBoard).

   Notez le port affiché par TensorBoard:
   
       TensorBoard 2.4.1 at http://0.0.0.0:PORT/ (Press CTRL+C to quit)

3. Créez un tunnel SSH. Dans un nouvel onglet **local**, exécutez ce qui suit (remplacez `PORT`):

       ssh -N -f -L localhost:PORT:login1:PORT <username>@ift780.calculquebec.cloud
       
   Note: cette commande ne retourne rien si tout va bien.
       
4. Vous pouvez ouvrir votre navigateur web à `localhost:PORT`.

## 4. Recherche d'hyperparamètres

Nous allons faire une recherche d'hyperparamètres très simple.

### Soumettre les tâches

Commençons un nouveau script à partir de l'ancien:

    cp train.sh hpsearch.sh
    
Ouvrez le nouveau script, faites l'ajout suivant à la ligne `python...`:

    python ~/atelier-dl-cc/train.py ./data --save-path $OUTDIR --epochs 10 --wd $HP_WEIGHT_DECAY

Finalement, vous pouvez lancer les différents essais comme suit:

    HP_WEIGHT_DECAY=0.0001 sbatch hpsearch.sh
    HP_WEIGHT_DECAY=0.001 sbatch hpsearch.sh
    HP_WEIGHT_DECAY=0.01 sbatch hpsearch.sh

### Comparer les choix d'hyperparamètres

1. Sur le noeud de login, allez dans le dossier qui contient les expériences à comparer, et vérifiez que les fichiers
`events.out.tfevents*` sont présents. Si `find` n'affiche rien, c'est mauvais signe.

       cd ~/projects/def-sponsor00/$USER/out
       find . -name 'events.out.tfevents*'

2. Utilisez TensorBoard pour faire une comparaison visuelle.

## Remarques

* Dans `train.sh`, ne mettez pas de ligne `salloc`.
* Lancez la commande `sbatch` sur un noeud de login, et non sur un noeud de calcul (obtenu avec un `salloc`).
* Vous pouvez lister vos tâches actives avec la commande `sq`, et annuler une tâche avec la commande `scancel <job_id>`.
    * "R" veut dire que la tâche est en exécution (running)
    * "PD" veut dire que la tâche est en attente (pending)
* Si vous obtenez l'erreur `Address already in use`, essayez de changer le port. Au lieu de 6006, essayez 6007, 6008,
  etc. Ensuite, changez `--port 6006` dans la commande `tensorboard`.
* La commande `ssh -N -f -L localhost ...` quitte tout de suite si elle fonctionne.
* Si en ouvrant Tensorboard, vous avez l'avertissement `Tensorflow not found`, vous pouvez l'ignorer.
* Si votre script `train.sh` contient moins de 10 commandes, relisez les instructions.
