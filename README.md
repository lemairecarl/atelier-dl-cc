# Atelier de Deep Learning sur Calcul Canada

## Se connecter au serveur

Ouvrez un terminal. (Sur Windows, vous aurez besoin d'installer MobaXTerm.)

       ssh <username>@helios3.calculquebec.ca

Dans cet environnement, vous pouvez:
* Préparer vos données
* Préparer votre code
* Préparer votre script de soumission
* Et tout autre opération, qui n'est pas l'entraînement.
      
Pour l'instant vous n'avez ni données, ni code. Nous allons régler ça dans la prochaine section.

## Transférer des données et du code

1. Téléchargez la bdd TinyImageNet à ce lien: https://drive.google.com/file/d/1g_MSfNMySQyhgqL8OIoP-nk3ogJCgWRM/view?usp=sharing
2. Transférez le fichier sur Hélios:

       rsync tinyimagenet.tar <username>@helios3.calculquebec.ca:

3. Déplacez le fichier dans l'espace de stockage "scratch". (Note: normalement, il faudrait transférer dans l'espace
   "projet", mais dans le cadre de cet atelier nous n'en avons pas.)
   
       ssh <username>@helios3.calculquebec.ca
       mv tinyimagenet.tar /scratch/<username>

4. Clonez le code dans votre `home` (`home/<username>`, c'est le dossier dans lequel vous arrivez lors d'une connexion
   `ssh`):

       git clone XXXXXXXXX

## Concevoir la séquence de commandes

À cette étape, il s'agit de trouver la bonne séquence de commandes qui permet d'effectuer correctement l'entraînement
sans supervision. Une fois cette séquence trouvée, on en fera un script (section suivante).

1. Soumission d'une tâche interactive. Demandez un GPU K20 et une heure:

       salloc --cpus-per-task=2 --gres=gpu:k20:1 --time=1:00:00



## Soumettre une tâche

sbatch

## Suivre le déroulement

tmux? ajouter au script sbatch?

    ssh <username>@helios3.calculquebec.ca
    ssh <noeud>
    tensorboard --logdir=$SLURM_TMPDIR/lightning_logs/version_4822 --host 0.0.0.0

où `SLURM_TMPDIR=/localscratch/<username>.<job_id>.0`

    ssh -N -f -L localhost:6006:<noeud>:6006 <username>@helios3.calculquebec.ca

## Récupérer les sorties de la tâche
