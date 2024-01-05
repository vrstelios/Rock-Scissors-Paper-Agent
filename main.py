import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

"""1) Match the moves (rock, scissors, paper) to (0, 1, 2 respectively)."""

pathData = 'your-path\\Rock-Scissors-Paper-Agent\\rps-cv-images'

moves = []
fileWithFigure = []
total_reward = 0
round_rewards = []

# Scan file rps_cv-images.
for folder in os.listdir(pathData):
    path = os.path.join(pathData, folder)

    if os.path.isdir(path):
        # Scan files paper, rock and scissors.
        for file in os.listdir(path):
            filePath = os.path.join(path, file)
            # Save the name of folder
            fileWithFigure.append(filePath)

            if folder.lower() == 'rock':
                moves.append(0)
            elif folder.lower() == 'paper':
                moves.append(1)
            elif folder.lower() == 'scissors':
                moves.append(2)

df = pd.DataFrame({'file figure': fileWithFigure, 'move': moves})
# print(df)

"""2) Divide the dataset into train-test: you can select a percentage for each class as For each class, you can select
      one test set for each class (e.g. 20% of the stone, 20% of the scissors and 20% of the hand) and assign a test set
      (e.g. 20% of the stone, 20% of the scissors and 20% of the hand) and then use these to test your model/agent."""

# Now we will split our data to train and test!
train_set, test_set = train_test_split(df, test_size=0.2, stratify=df['move'], random_state=0)

# print("Train set dimensions:", train_set.shape)
# print("Test set dimensions:", test_set.shape)

"""3) Select Image: randomly select an image from the 2100 (corresponding to either 0, 1, or 2)."""


def randomSelectImage(images_per_move=700):
    # random settlement movements (0, 1, ή 2)
    move = np.random.choice([0, 1, 2])
    moveImage = df[df['move'] == move].sample(images_per_move, replace=True)
    # random image selection
    selectedImage = moveImage.sample(1)

    return selectedImage['file figure'].values[0], move


selectedImage, move = randomSelectImage()

# print("Selected Image:", selectedImage)
# print("Corresponding Move:", move)

"""4)Preprocess Image: Edit the image:
     a. With probability p applied Vertical Flip.
     b. With probability p applied Horizontal Flip.
     c. Add noise with m = 0, σ = 255 * 0. 05). If you normalize the image,
     adjust the standard deviation accordingly.
     d. You can apply any other Image Processing method you want, in order to
     make the game more difficult (if you wish)."""


def preprocessImage(imagePath, targetSize, pFlip, pHorizontalFlip, noise_mean, noise_std, brightness_factor,
                    apply_filter):
    # Reading an image from a file
    image = cv2.imread(imagePath)
    # Dimensionality Reduction of image
    image = cv2.resize(image, targetSize)

    # Dimensionality Reduction:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalization
    image = image / 255.0

    # Adjust brightness
    image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

    # Apply filter if specified
    if apply_filter:
        image = cv2.GaussianBlur(image, (5, 5), 0)  # Example: Gaussian Blur

    # Vertical Flip application with probability pFlip
    if np.random.rand() < pFlip:
        image = cv2.flip(image, 0)

    # Horizontal Flip application with probability pHorizontalFlip
    if np.random.rand() < pHorizontalFlip:
        image = cv2.flip(image, 1)

    # Add noise to the image
    noise = np.random.normal(noise_mean, noise_std, image.shape)
    scaled_noise = noise * 0.00001  # Adjusting the amount of noise

    # Clip values to [0, 1] after adding noise
    noisy_image = np.clip(image + scaled_noise, 0, 1)

    return noisy_image


# preprocessedImage = preprocessImage(selectedImage, (30, 30), 0.5, 0.5, 0, 255 * 0.05, 1.0, False)

# Show the edited image and the corresponding motion
# cv2.imshow("Preprocessed Image", preprocessedImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print("Corresponding Move:", move)

"""5) Apply: send the image to your agent."""
"""6) The agent reads the image and chooses the optimal action."""

# Pre-processing of training images.
X_train = np.array([preprocessImage(imagePath, (30, 30), 0.5, 0.5, 0, 255 * 0.05, 1.0, False) for imagePath in
                    train_set['file figure']])
X_train = X_train.reshape(X_train.shape[0], -1)

# Normalisation of training data.
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)

# Creation and training of the model MLPClassifier.
# mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=0)
# mlp_classifier.fit(X_train, train_set['move'])

# Creation and training of the model KNN
# knn_classifier = KNeighborsClassifier(n_neighbors=3)
# knn_classifier.fit(X_train, train_set['move'])

# Definition of the model CNN
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(30, 30, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Adapting the model for training
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

# Definition of the training process
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the model with the training data
X_train_cnn = X_train.reshape((X_train.shape[0], 30, 30, 1))
model.fit(X_train_cnn, train_set['move'], epochs=10, batch_size=32, validation_split=0.2)


def get_agent_action(image):
    preproImage = preprocessImage(image, (30, 30), 0.5, 0.5, 0, 255 * 0.05, 1.0, False)

    # Application to the model for prediction
    # preprocessedImage_flattened = preproImage.reshape(1, -1)
    # preprocessedImage_scaled = scaler.transform(preprocessedImage_flattened)
    # prediction = mlp_classifier.predict(preprocessedImage_scaled)
    # prediction = knn_classifier.predict(preprocessedImage_scaled)
    preprocessedImage_cnn = preproImage.reshape((1, 30, 30, 1))
    prediction = model.predict(preprocessedImage_cnn)

    return np.argmax(prediction)
    # return prediction[0]


#   Add if-else logic for action selection
#   if prediction[0] == 0:
#        return "Rock"
#   elif prediction[0] == 1:
#        return "Paper"
#   elif prediction[0] == 2:
#        return "Scissors"
#   else:
#        return "Unknown"""""


# agent_action = get_agent_action(selectedImage)
# print("Agent's Action:", agent_action)

"""7) The aim is to maximise profit. So, you will need to plot the profit of the
   (You can save the total profit in each round and plot it at the end of the game)."""

totalReward = 0
roundRewards = []
countLose = 0
countWin = 0
N = 100
for _ in range(N):
    winAgent = False
    loseAgent = False
    currentRoundReward = 1

    while not (winAgent or loseAgent):
        # Random motion selection for the Random Agent
        imageRandomAgent, moveRandomAgent = randomSelectImage()
        # Movement of the agent
        moveAgent = get_agent_action(imageRandomAgent)
        # bet
        currentRoundReward -= 1

        # Check for which move wins
        if (moveRandomAgent + 1) % 3 == moveAgent:
            currentRoundReward += 2
            countWin += 1
            winAgent = True
        elif moveRandomAgent == moveAgent:
            currentRoundReward += 1
        else:
            loseAgent = True
            countLose += 1

        # print(currentRoundReward, winAgent, loseAgent)
        totalReward += currentRoundReward
        roundRewards.append(totalReward)

# print(totalReward)
print("winning rate", (countWin / N) * 100)
print("Rate of defeats", (countLose / N) * 100)

# show plot
plt.plot(roundRewards)
plt.xlabel('Round')
plt.ylabel('Total Profit')
plt.title('Total Profit per Round')
plt.show()

'''8) As an end goal, test the accuracy of your agent (or model) on images outside of the dataset,
   e.g. from the internet or your own, which you should rescale to the same dataset size.'''

# Load the image
new_image_path = ('your-path\\Rock-Scissors-Paper-Agent\\rps-cv-images\\scissors'
                  '\\0CSaM2vL2cWX6Cay.png')

# Pre-processing of the new image
preprocessed_new_image = preprocessImage(new_image_path, (30, 30), 0.5, 0.5, 0, 255 * 0.05, 1.0, False)

# Scaling of the pre-processed image
# preprocessed_new_image_flattened = preprocessed_new_image.reshape(1, -1)
# scaled_new_image = scaler.transform(preprocessed_new_image_flattened)

preprocessedImage_cnn = preprocessed_new_image.reshape((1, 30, 30, 1))
prediction_for_new_image = model.predict(preprocessedImage_cnn)

# Running the model for the new image
# prediction_for_new_image = mlp_classifier.predict(scaled_new_image)
# prediction_for_new_image = knn_classifier.predict(scaled_new_image)

# Printing the forecast
# if prediction_for_new_image[0] == 0:
#     print("Predicted Move - Rock")
# elif prediction_for_new_image[0] == 1:
#     print("Predicted Move - Scissors")
# elif prediction_for_new_image[0] == 2:
#     print("Predicted Move - Paper")
if np.argmax(prediction_for_new_image) == 0:
    print("Predicted Move - Rock")
elif np.argmax(prediction_for_new_image) == 1:
    print("Predicted Move - Scissors")
elif np.argmax(prediction_for_new_image) == 2:
    print("Predicted Move - Paper")

''' Απαντήσεις '''

'''Πειραματιστείτε με διάφορα μοντέλα, μεθόδους επεξεργασίας εικόνων, τεχνικές
   βελτίωσης της ακρίβειας των μοντέλων/πρακτόρων κλπ. Μπορείτε να υποβάλλετε 
   την άσκηση με τη χρήση μόνο ενός μοντέλου/πράκτορα, αλλά να αιτιολογήσετε 
   το λόγο που το επιλέξατε (πχ το επελεξα γιατί φαίνεται πως έχει απόδοση 99%).
   
   Convolutional Neural Networks (CNN): οφείλεται στην ικανότητά του να αντιλαμβάνεται
   την δομή των εικόνων και να αντιμετωπίζει αποτελεσματικά προβλήματα όπως η αναγνώριση
   αντικειμένων και στη συγκεκριμένη περίπτωση, όπως το παιχνίδι περτα-ψαλίδι-χαρτί

   Σε σύγκριση με άλλους ταξινομητές όπως ο MLP (Multi-Layer Perceptron) ή ο 
   K-Nearest Neighbors (KNN), που είναι πιο κατάλληλοι για δομές δεδομένων όπως πίνακες,
   τα CNN έχουν την δυνατότητα να αντιληφθούν πρότυπα και χαρακτηριστικά που βρίσκονται 
   σε κοντινή γειτονιά εικονοστοιχείων, κάτι που τους καθιστά ιδανικούς για προβλήματα 
   επεξεργασίας εικόνων.

   Συνοπτικά, η απόδοση του CNN με ποσοστό νίκης 60% φαίνεται ικανοποιητική σε σχέση με 
   τους άλλους ταξινομητές, κάνοντάς τον κατάλληλη επιλογή για το πρόβλημα που 
   αντιμετωπίζετε.'''

'''Κάντε ανάλυση σε ποιες εικόνες δουλεύει καλά και σε ποιες όχι η τελική σας λύση, καθώς
   και με ποιες τεχνικές το αντιμετωπίσατε.
     
   Ο Convolutional Neural Networks (CNN) σύμφωνα με την ανάλυση που έκανα δουλεύει εξαιρετικά
   στην εικόνα Rock, στην εικόνα scissors λίγες φορές το μπερδεύει με paper κύριος όταν η 
   φιγούρα από το μαύρο χρώμα δεν φαίνονται καθαρά τα δάχτυλα και τέλος τα paper οπού τα 
   μπερδεύει αρκετά εύκολα με εικόνες scissors πάλι με το ίδιο πρόβλημα όπως με τα scissors 
   άπλα εδώ γίνεται ποιο συχνά.

   Σίγουρα η κύρια παράμετρος που επηρεάζει την απόδοση του μοντέλου είναι η ποσότητας 
   θορύβου επηρεάζει πολύ την απόδοση του μοντέλου η καλύτερη λύσει για την δικιά μου 
   υλοποίηση είναι 0,00001 άλλα και το ποσό της φωτεινότητας παίζει ρόλο για την αντίθεση 
   του μαύρο και του άσπρου χρώματος για να ξεχωρίσει καλύτερα η φιγούρα.'''

'''Αιτιολογήστε τα μοντέλα και τις τεχνικές που επιλέξατε, καθώς και την απόδοση σας.

   Για Ν = 100
   Convolutional Neural Network (CNN)
   winning rate 60.0, Rate of defeats 40.0

   Multi-Layer Perceptron (MLP)
   winning rate 54.0, Rate of defeats 46.0
   
   K-Nearest Neighbors (KNN):
   winning rate 41.0, Rate of defeats 59.0

    Convolutional Neural Networks (CNN): Χρήση: Ιδανικό για εικόνες. Εξαιρετικά 
    αποτελεσματικό για εξαγωγή χαρακτηριστικών και ταξινόμηση.

    K-Nearest Neighbors (KNN) Χρήση: Κατάλληλο για ταξινόμηση, αλλά μπορεί να είναι
    υπολογιστικά απαιτητικό για μεγάλα σύνολα δεδομένων.

   Multi-Layer Perceptron (MLP) Χρήση: Κατάλληλο για ποικίλα προβλήματα μη γραμμικής
   κατηγοριοποίησης και παλινδρόμησης. Είναι ευέλικτο και κατάλληλο για προβλήματα όπου
   οι σχέσεις μεταξύ των χαρακτηριστικών είναι πολύπλοκες.'''