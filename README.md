# Energy data hack
## Challenge 1 : Interception des ondes radios sur les capteurs d’énergies

<p align="center">

<a >
    <img src='./images/logo_hackaton.png'  width="400"/>
</a>

</p>

Hacker le login et le mdp sur une interception d'ondes radios. 

## Equipes 


- `Maxence Brugères (Telecom Paris & Sciences Po)` :  Classification random forest, outil de séparation de trames en touche, outil de production de mots
- `Clément Bernard (Telecom Paris & Polytechnique Montréal)` : Classification GRU, MLP, outil de séparation des touches du login et du mot de passe, mco du répo Github
- `Inès Benito (ENSTA Bretagne)` :  Classification via d’autre modèle (Logistic regression, KNN), recherches théoriques et appliquées pour entrainer des modèles avec des loss custom, synthèse des travaux pour le rapport, data visualisation
- `Corentin Lestrat (ENSTA Bretagne)` : État de l’art des exploitations de SPC avec du machine learning, recherche biblio sur des sujets divers en support sachant qu’il n’avait aucune formation technique en machine learning, social engineering




## Installation

Utiliser un 'venv' ou un environnement 'conda', puis récupérer les versions :
```
pip install -r requirements.txt
```

## Entrainement des modèles

- Utilisation de MLP, GRU et Random Forest pour la classification multi-classe (42 classes)

| Accuracy | F1 score |
|---| --- |
| ![](/images/acc.png) | ![](/images/f1_scores.png) |

Les modèles MLP et GRU sont plus précis que Random Forest sur les données d'entrainement et de validation.


- Utilisation de MLP et GRU pour données regroupées en groupe de trames : 1, 5, 10, 25, 100 et 200

| MLP | GRU |
|---| --- |
| ![](/images/acc_without_noise_mlp.png) | ![](/images/acc_without_noise_gru.png) |

Plus on regroupe par trame, et meilleure est l'accuracy. 


- Utilisation de MLP et GRU pour données regroupées en groupe de trames avec ajout de bruit sur les données d'entrainement

| MLP | GRU |
|---| --- |
| ![](/images/acc_noise_mlp.png) | ![](/images/acc_noise_gru.png) |

Les regroupements par trame avec bruit est efficace pour un regroupement faible, et pas pour un groupe de trames élevé


## Prédictions sur l'ensemble de test : les données d'un utilisateur


- Prédictions trame par trame, avec les probabilités d'être dans un label :

| MLP | GRU | Random Forest |
|---| --- | --- | 
| ![](/images/y_test_proba_mlp.png) | ![](/images/y_test_proba_gru.png) |![](/images/y_test_proba_rf.png)  |

Les modèles MLP et GRU sont plus précis et donc ont du mal à reconnaitre les différences lorsque plusieurs touches sont appuyés en même temps.
Random Forest est moins bon et donc est plus ambigue : il peut donc voit lorsque deux touches sont appuyées en même temps


- Prédictions par groupe de trames sans bruit, avec GRU par exemple : 


| GRU |
| --- | 
| ![](/images/y_test_gru_stack.png) |

Plus on regroupe par trame, et plus on réduit le bruit, mais on perd de la précision et n'arrivons pas à distinguer entre plusieurs touches pressées en même temps. 



- Prédictions par groupe de trames avec bruit, avec GRU par exemple : 


| GRU avec bruit |
| --- | 
| ![](/images/y_test_gru_noise.png) |


- On retient finalement que la prédiction avec Random Forest trame par trame car le modèle classifie moins bien mais donne plus de données sur les probabilités de chaque caractère. 


## Décodage du login et mot de passe 

- On utilise la matrice de confusion sur l'ensemble de validation de Random forest pour prédire les caractères sur plusieurs trames (minimisation de la distance entre les lignes de la matrice de confusion et les probabilités de prédiction de Random Forest). 



| Matrice de confusion | Prédictions de Random Forest |
|---| --- |
| ![](/images/confusion_matrix.png) | ![](/images/y_test_confusion.png) |



- Identification des trames appartenant à une même touche pressée. Utilisation d'une fenêtre glissante sur 30 trames pour lisser les probabilités, puis on moyenne la distance euclidienne entre chacun des vecteurs de probabilités.  

| Identification des trames d'une même touche | 
|---| 
| ![](/images/pics.png) | 


- Utilisation d'un algorithme de clustering pour différencier 3 groupes : 1) déverouillage 2) login et 3) mot de passe. 

| Clustering en 3 groupes  | 3 groupes obtenus |
| --- | --- | 
| ![](/images/clustering.png) | ![](/images/clustering_groupes.png) | 

- Distance par rapport aux touches connues (avec la matrice de confusion) pour chaque touche pressée

| Clustering en 3 groupes  |
| --- | 
| ![](/images/distance_touches_connues.png) | 



- Identification des caractères : on considère pour chaque groupe de trames les caractères les plus probables.


| Prédictions des caractères  |
| --- | 
| ![](/images/caracteres.png) | 




## Résultats 

- La prédiction des logins et de mot de passe possible se trouve dans : 
 - `predictions`
Il y a un fichier `login.csv` et `mdp.csv` avec les prédictions des logins et des mdp les plus probables. 
- Nous pensons que le login est le suivant : **DGAMI** et le mot de passe : **hackaton2021**





## Dépo 

- `Hackaton` : Fichiers avec les données du challenge
- `images` : Images utilisées dans le README
- `models` : Les modèles enregistrés lors de training 
- `models_noise` : Les modèles entrainés avec du bruit 
- `models_stack` : Les modèles entrainés sans bruit et par groupement de trames 
- `prédictions` : Les prédictions du login et mot de passe 
- `python` : Fichiers python utilisés 
- `Hackaton.zip` : Fichier zip de `Hackaton` 
- `models.zip`: Fichier zip de `models`
- `models_noise.zip`: Fichier zip de `models_noise`
- `models_stack.zip`: Fichier zip de `models_stack`
- `python.zip`: Fichier zip de `python`
- `requirements.txt` : Ficher avec les requirements sous format txt
 






