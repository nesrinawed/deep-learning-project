from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
# Charger l'ensemble de données ORL
donnees = fetch_olivetti_faces()
# Accéder aux attributs de données et de cible
images = donnees.images
cibles = donnees.target
# Vérifier la forme des données
print("Forme du tableau d'images :", images.shape)
print("Nombre d'échantillons :", len(images))
print("Nombre de cibles :", len(cibles))
# Définir une fonction pour afficher plusieurs images
def afficher_images(images, cibles, num_lignes=3, num_colonnes=5):
    fig, axes = plt.subplots(num_lignes, num_colonnes, figsize=(10, 6),
                             subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    # Sélectionner un sous-ensemble aléatoire d'images
    num_images = num_lignes * num_colonnes
    indices = np.random.choice(len(images), size=num_images, replace=False)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[indices[i]], cmap='gray')
        ax.set_title(f"Personne {cibles[indices[i]]}")
    plt.show()
# Afficher les images
afficher_images(images, cibles)
# Aplatir chaque image en une seule dimension
nombre_echantillons, hauteur, largeur = images.shape
images_aplaties = images.reshape(nombre_echantillons, hauteur * largeur)
# Normalisation des valeurs de pixel
normaliseur = StandardScaler()
images_normalisees = normaliseur.fit_transform(images_aplaties)
# Appliquer l'ACP pour réduire la dimensionnalité
acp = PCA(n_components=30)  # Vous pouvez ajuster le nombre de composantes selon vos besoins
images_reduites = acp.fit_transform(images_normalisees)
# Reconstruire les images à partir des données réduites
images_reconstruites = acp.inverse_transform(images_reduites).reshape(nombre_echantillons, hauteur, largeur)
# Afficher les images reconstruites
afficher_images(images_reconstruites, cibles)
###CNN
# Charger les données
images = images_reconstruites
# Diviser les données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(images, cibles, test_size=0.2, random_state=42)
# Redimensionner les données pour qu'elles correspondent à l'entrée du modèle CNN
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
# Créer le modèle CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(40, activation='softmax')  # 40 classes pour les 40 visages dans le jeu de données
])
# Compiler le modèle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Entraîner le modèle
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
# Évaluer le modèle
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")
# Afficher les courbes d'apprentissage
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()