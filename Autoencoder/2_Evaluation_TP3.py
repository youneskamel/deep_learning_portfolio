# La libraire responsable du chargement des données dans la mémoire
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator

# Affichage des graphes et des images
import matplotlib.pyplot as plt

# La librairie numpy
import numpy as np

# Configuration du GPU
import tensorflow as tf

# Utlilisé pour charger le modèle
from keras.models import load_model
from keras import Model

# Utilisé pour normaliser l'embedding
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from keras import backend as K
from sklearn.manifold import TSNE

# Configuration des GPUs et CPUs
config = tf.compat.v1.ConfigProto(device_count={'GPU': 1, 'CPU': 16})
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


# Chargement du modéle (autoencodeur) sauvegardé dans la section 1 via 1_Modele.py
model_path = "Model.hdf5"
autoencoder = load_model(model_path)

# L'emplacement des images
mainDataPath = "vache_elephant/"

# On évalue le modèle sur les images tests
datapath = mainDataPath + "test"

# Le nombre des images de test à évaluer
number_images = 400  # 400 images
number_images_class_0 = 200  # 200 images pour la classe de l'éléphant
number_images_class_1 = 200  # 200 images pour la classe de la vache

# Les étiquettes (classes) des images
labels = np.array([0] * number_images_class_0 +
                  [1] * number_images_class_1)

# Le dossier contenant les images d'entrainement
trainPath = mainDataPath + "entrainement"

# Le nombre d'images d'entrainement
training_ds_size = 2400

# La taille des images
image_scale = 140

# La couleur des images
images_color_mode = "rgb"  # grayscale ou rgb

# Le nombre de canaux de couleurs
image_channels = 3

# Chargement des images test
data_generator = ImageDataGenerator(rescale=1. / 255)

generator = data_generator.flow_from_directory(
    datapath,  # Place des images d'entrainement
    color_mode=images_color_mode,  # couleur des images
    target_size=(image_scale, image_scale),  # taille des images
    batch_size=number_images,  # nombre d'images total à charger en mémoire
    class_mode=None,
    shuffle=False)  # pas besoin de bouleverser les images

x = generator.next()

decoded = autoencoder.predict(x)

for i, j, label in ((2, 1, 'Éléphant'), (200, 2, 'Vache')):
    # Display original
    ax = plt.subplot(2, 2, j)
    plt.imshow(x[i])
    ax.set_title(label)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, 2, j + 2)
    plt.imshow(decoded[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# Défintion du modèle
autoencoder.summary()
input_layer_index = 0  # l'indice de la première couche de l'encodeur (input)
output_layer_index = 8  # l'indice de la dernière couche (la sortie) de l'encodeur (dépend de votre architecture)
# note: Pour identifier l'indice de la dernière couche de la partie encodeur, vous pouvez utiliser la fonction "model.summary()"
# chaque ligne dans le tableau affiché par "model.summary" est compté comme une couche
encoder = Model(autoencoder.layers[input_layer_index].input, autoencoder.layers[output_layer_index].output)

encoded = encoder.predict(x)

image_shape = (image_scale, image_scale,
               image_channels)  # la forme des images d'entrées, ce qui correspond à la couche d'entrée du réseau

encoded = Flatten(input_shape=image_shape)(encoded)
x = Flatten(input_shape=image_shape)(x)

scaler = StandardScaler()
encoded = scaler.fit_transform(encoded)

def split_ratio(list_to_split, ratio):
    elements = len(list_to_split)
    middle = int(elements * ratio)
    return [list_to_split[:middle], list_to_split[middle:]]


x_train, x_test = split_ratio(x, 0.8)
labels_train, labels_test = split_ratio(labels, 0.8)

model = svm.SVC(kernel='linear', probability=True,
                C=1)  # Utiliser probability=true permet l'utilisation de la cross-validation
model.fit(x_train, labels_train)
pred = model.predict(x_test)
print(f"Accuracy of SVM Linear on original images {accuracy_score(pred, labels_test)}")


encoded_train, encoded_test = split_ratio(encoded, 0.8)

model = svm.SVC(kernel='linear', probability=True,
                C=1)  # Utiliser probability=true permet l'utilisation de la cross-validation
model.fit(encoded_train, labels_train)
pred = model.predict(encoded_test)
print(f"Accuracy of SVM Linear on encodings {accuracy_score(pred, labels_test)}")


plt.clf()
tsne_proj = TSNE(n_components=2).fit_transform(encoded)
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(tsne_proj[:,0],tsne_proj[:,1], c=labels)
plt.show()