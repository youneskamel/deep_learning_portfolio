
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, UpSampling2D, Activation, Dropout, Flatten, \
    Dense
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import backend as K
import time
config = tf.compat.v1.ConfigProto(device_count={'GPU': 1, 'CPU': 16})
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess);

# Le dossier principal qui contient les données
mainDataPath = "vache_elephant/"

# Le dossier contenant les images d'entrainement
trainPath = mainDataPath + "entrainement"

# Le dossier contenant les images de validation
validationPath = mainDataPath + "entrainement"

# Le nom du fichier du modèle à sauvegarder
model_path = "Model.hdf5"

# Le nombre d'images d'entrainement
training_ds_size = 2000  # total 2000 (1000 classe: 2 et 1000 classe: 7)
validation_ds_size = 400  # total 400 (200 classe: 2 et 200 classe: 7)

# Configuration des  images
image_scale = 140  # la taille des images
image_channels = 3  # le nombre de canaux de couleurs (1: pour les images noir et blanc; 3 pour les images en couleurs (rouge vert bleu) )
images_color_mode = "rgb"  # grayscale pour les image noir et blanc; rgb pour les images en couleurs
image_shape = (image_scale, image_scale,
               image_channels)  # la forme des images d'entrées, ce qui correspond à la couche d'entrée du réseau

# Configuration des paramètres d'entrainement
fit_batch_size = 32  # le nombre d'images entrainées ensemble: un batch
fit_epochs = 100  # Le nombre d'époques

# Couche d'entrée:
# Cette couche prend comme paramètre la forme des images (image_shape)
input_layer = Input(shape=image_shape)


# Partie d'encodage (qui extrait les features des images et les encode)
def encoder(input):
    x = Conv2D(32, (3, 3), padding='same')(input)
    x = Activation('sigmoid')(x)
    x = Dropout(0.25)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.25)(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # encoded : La sortie de l'encodeur consistue l'embedding (ou les descripteurs extraites par l'encodeur)
    return encoded


# Partie de décodage (qui reconstruit les images à partir de leur embedding ou la sortie de l'encodeur)
def decoder(encoded):
    x = Conv2D(64, (3, 3), padding='same')(encoded)
    x = Activation('sigmoid')(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = Activation('sigmoid')(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(image_channels, (3, 3), padding='same')(x)

    decoded = Activation('sigmoid')(x)
    return decoded


# Déclaration du modèle:
model = Model(input_layer, decoder(encoder(input_layer)))
model.summary()

# Compilation du modèle
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

# Chargement des données
training_data_generator = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    fill_mode='nearest',
    width_shift_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.4, 1.5],
    validation_split=0.2
)

validation_data_generator = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

training_generator = training_data_generator.flow_from_directory(
    trainPath,  # Place des images d'entrainement
    color_mode=images_color_mode,  # couleur des images
    target_size=(image_scale, image_scale),  # taille des images
    batch_size=training_ds_size,  # nombre d'images total à entrainer
    subset='training',
    class_mode="input")  

validation_generator = validation_data_generator.flow_from_directory(
    validationPath,  # Place des images d'entrainement
    color_mode=images_color_mode,  # couleur des images
    target_size=(image_scale, image_scale),  # taille des images
    batch_size=validation_ds_size,  # nombre d'images total à valider
    subset='validation',
    class_mode="input")  

# On charge les données d'entrainement en mémoire
(x_train, _) = training_generator.next()
# On charge les données de validation en mémoire
(x_val, _) = validation_generator.next()

# Entrainement
modelcheckpoint = ModelCheckpoint(filepath=model_path,
                                  monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

# entrainement du modèle
start_time = time.perf_counter()
autoencoder = model.fit(x_train, x_train,
                        epochs=fit_epochs,  # nombre d'epochs
                        batch_size=fit_batch_size,  # nombre d'images entrainées ensemble
                        verbose=1,  
                        callbacks=[modelcheckpoint],
                        # les fonctions à appeler à la fin de chaque epoch (dans ce cas modelcheckpoint: qui sauvegarde le modèle)
                        shuffle=False,  # On ne boulverse pas les données
                        validation_data=(x_val, x_val))  # données de validation
end_time = time.perf_counter()


print("Temps d'execution :", end_time - start_time)

plt.plot(autoencoder.history['loss'])
plt.plot(autoencoder.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
