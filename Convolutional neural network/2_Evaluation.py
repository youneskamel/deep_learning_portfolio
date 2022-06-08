# **************************************************************************
# INF7370 - Apprentissage automatique - Hiver 2022
# Travail pratique 2
# Younes Kamel(KAMY15029708)
# Maxime Nicol (NICM18019305)
# ===========================================================================

#===========================================================================

# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

# La libraire responsable du chargement des données dans la mémoire
from keras.preprocessing.image import ImageDataGenerator

# Affichage des graphes
import matplotlib.pyplot as plt

# La librairie numpy 
import numpy as np

# Configuration du GPU
import tensorflow as tf
from keras import backend as K

# Utilisé pour le calcul des métriques de validation
from sklearn.metrics import confusion_matrix, roc_curve , auc

# Utlilisé pour charger le modèle
from keras.models import load_model
from keras import Model

# Affichage des images
from PIL import Image, ImageDraw, ImageFont
from mpl_toolkits.axes_grid1 import ImageGrid

# ==========================================
# ===============GPU SETUP==================
# ==========================================

# Configuration des GPUs et CPUs
config = tf.compat.v1.ConfigProto(device_count={'GPU': 2, 'CPU': 4})
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess);

# ==========================================
# ==================MODÈLE==================
# ==========================================

#Chargement du modéle sauvegardé dans la section 1 via 1_Modele.py
model_path = "Model.hdf5"
Classifier: Model = load_model(model_path)

# ==========================================
# ================VARIABLES=================
# ==========================================

# Répertoire où sauvegarder les photos mal étiquetées
out_path = "out/"


# Dictionnaire des indices de classes 
class_dict = {'elephant': 0, 'girafe': 1, 'leopard': 2, 'rhino': 3, 'tigre': 4, 'zebre': 5}


# Liste des classes
classes = [name for name,_ in class_dict.items()]

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                       QUESTIONS
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 1) A ajuster les variables suivantes selon votre problème:
# - mainDataPath         
# - number_images        
# - number_images_class_x
# - image_scale          
# - images_color_mode    
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# L'emplacement des images de test
mainDataPath = "donnees/"
testPath = mainDataPath + "test"

# Le nombre des images de test à évaluer
number_images = 6000

# 1000 images pour chacune des 6 classes
number_images_class_0 = 1000
number_images_class_1 = 1000
number_images_class_2 = 1000
number_images_class_3 = 1000
number_images_class_4 = 1000
number_images_class_5 = 1000

# La taille des images à classer
image_scale = 140

# La couleur des images à classer
images_color_mode = "rgb"  # grayscale or rgb

# ==========================================
# =========CHARGEMENT DES IMAGES============
# ==========================================

# Chargement des images de test
test_data_generator = ImageDataGenerator(rescale=1. / 255)

test_itr = test_data_generator.flow_from_directory(
    testPath,# place des images
    target_size=(image_scale, image_scale), # taille des images
    class_mode="categorical",# Type de classification
    shuffle=False,# pas besoin de les boulverser
    batch_size=1,# on classe les images une à la fois
    color_mode=images_color_mode)# couleur des images

(x, y_true) = test_itr.next()

# Normalize Data
max_value = float(x.max())
x = x.astype('float32') / max_value

# ==========================================
# ===============ÉVALUATION=================
# ==========================================

# Les classes correctes des images (1000 pour chaque classe) -- the ground truth
y_true = np.array([0] * number_images_class_0 + 
                  [1] * number_images_class_1 + 
                  [2] * number_images_class_2 + 
                  [3] * number_images_class_3 + 
                  [4] * number_images_class_4 + 
                  [5] * number_images_class_5)

# evaluation du modËle
test_eval = Classifier.evaluate_generator(test_itr, verbose=1)

# Affichage des valeurs de perte et de precision
print('>Test loss (Erreur):', test_eval[0])
print('>Test précision:', test_eval[1])

# Prédiction des classes des images de test
predicted_classes = Classifier.predict_generator(test_itr, verbose=1)
predicted_classes_perc = np.round(predicted_classes.copy(), 4)
predicted_classes = np.round(predicted_classes, 2) # on arrondie le output
predicted_class_indices = np.argmax(predicted_classes, axis=1)
# 0 => classe elephant
# 1 => classe girafe
# 2 => classe leopard
# 3 => classe rhino
# 4 => classe tigre
# 5 => classe zebre

# Cette list contient les images bien classées
correct = []
for i in range(0, len(predicted_classes) ):
    if predicted_class_indices[i] == y_true[i]:
        correct.append(i)

# Nombre d'images bien classées
print("> %d  Étiquettes bien classÈes" % len(correct))

# Cette list contient les images mal classées
incorrect = []
for i in range(0, len(predicted_classes) ):
    if predicted_class_indices[i] != y_true[i]:
        incorrect.append(i)

# Nombre d'images mal classées
print("> %d Ètiquettes mal classÈes" % len(incorrect))

# ***********************************************
#                  QUESTIONS
# ***********************************************
# 1) Afficher la matrice de confusion
# 2) Extraire une image mal-classée pour chaque combinaison d'espèces - Voir l'exemple dans l'énoncé.
# ***********************************************

# ####################################################################
# 1) Afficher la matrice de confusion 
# ####################################################################

# Génération de la matrice de confusion
conf_mx = confusion_matrix(predicted_class_indices, y_true)

# Création du squelette général matplotlib
fig = plt.figure()
ax = fig.add_subplot(111)

# Création du graphique utilisant le cmap "Blues"
cax = ax.matshow(conf_mx, cmap=plt.cm.get_cmap("Blues"))
fig.colorbar(cax)

# Ajout des valeurs de cellules
for i in range(0, 6):
    for j in range (0,6):
        ax.text(j,i, "{:0.1f}".format(conf_mx[j][i]), ha="center", va="center")

# Déplacement de l'axe des x en dessous du graphique
ax.xaxis.set_ticks_position("bottom")
ax.xaxis.set_label_position("bottom")

# Remplacement des indexes des axes par la classe correspondante
ax.set_xticklabels([''] + classes, rotation=45)
ax.set_yticklabels([''] + classes)

# Ajout des titres des axes
ax.set_ylabel("true label", rotation='vertical',)
ax.set_xlabel("predicted label", rotation='horizontal')

# Affichage du graphique
plt.savefig("conf_matrice.png")

# ####################################################################
# 2) Extraire une image mal-classée pour chaque combinaison d'espèces 
# ####################################################################

def find_match(reference, predicted):
    pred_idx = class_dict[predicted]
    ref_idx = class_dict[reference]
    for idx in range(0, len(predicted_classes)):
        if (predicted_class_indices[idx] == pred_idx) and (y_true[idx] == ref_idx):
            return idx

# Création de la structure de base ImageGrid
plt.rcParams.update({'font.size': 48})
fig = plt.figure(figsize=(36.,36.))
grid = ImageGrid(fig, 111,
                nrows_ncols=(6,6),
                axes_pad=0.1,
                )
images = []

for ref,r_idx in class_dict.items():
    for found,f_idx in class_dict.items():
        im_idx = find_match(ref, found)
        if im_idx is not None :
            filename = f"{testPath}/{test_itr.filenames[im_idx]}"
            with open(filename,'rb') as f:
                image = Image.open(f)
                images.append(image.resize((300,300), Image.ANTIALIAS))
        else :
            img = Image.new('RGB', (300, 300), color = (128, 128, 128))
            images.append(img)

for ax, im in zip(grid, images):
    if (im is not None):
        ax.imshow(im)
        ax.set_xticks([], minor=False)
        ax.set_yticks([], minor=False)

# Ajout des titres de colonnes
for ax, col in zip(grid[0:6], classes) :
    ax.set_title(col)

# Ajout des titres de rangées
for ax, row in zip(grid[::6], classes):
    ax.set_ylabel(row, rotation='horizontal', size='large')


# Affichage de la table
plt.savefig("combinaisons.png")
