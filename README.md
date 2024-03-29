﻿# Rock-Scissors-Paper-Agent

The goal of this project is to build an intelligent agent that learns to play the Rock-Scissors-Paper game. Specifically, the agent receives an image corresponding to 0: Rock, 1: Scissors, or 2: Paper and chooses the corresponding symbol that beats it.

# Description of Rock-Scissors-Paper-Agent
Rock-Scissors-Paper-Agent is a model trained to recognize images provided as input and respond with another image with the aim of winning in the Rock-Scissors-Paper game.

# How it Works
Our model reads all images (Rock-Scissors-Paper) from the [dataset](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors) separates the data into training and test sets, and implements a function called randomSelectImage() to randomly select an image from our dataset. This function is used when the model needs to play against a random opponent in the Rock-Scissors-Paper game. Additionally, we have implemented a function called preprocessImage() to process the images. In this function, we reduce the pixel density of the images, resize them to 30x30, convert them to grayscale, and normalize the pixel values to the range [0, 1]. We also introduce challenges such as noise, vertical flip, and horizontal flip to simulate real-world scenarios. After randomly selecting and processing the image, it's time to train the model.

The model is trained using the get_agent_action function. In this function, three models are implemented: MLPClassifier, KNN, and CNN. However, we choose to keep only the CNN model, as it has the highest success rate among the models. With the CNN model, we have all the tools we need. The next step is to play the randomSelectImage() and get_agent_action() functions and measure the score achieved by the trained player. Finally, we test the model with specific images to discover which type of image (Rock-Scissors-Paper) the model struggles to recognize.

Feel free to customize and adapt the above text to better fit your specific project details.

# How to Run
Make sure to replace the values of pathData and new_image_path with your specific dataset path before running the main.py file.

Python Version & Libraries
- **Python:** [3.7](https://www.python.org/downloads/release/python-370/)
- **OpenCV:** [cv2](https://pypi.org/project/opencv-python/)
- **NumPy:** [NumPy](https://numpy.org/)
- **Pandas:** [Pandas](https://pandas.pydata.org/)
- **TensorFlow:** [2.9.1](https://www.tensorflow.org/install)
- **Matplotlib:** [Matplotlib](https://matplotlib.org/)
- **scikit-learn:** [scikit-learn](https://scikit-learn.org/stable/install.html)

# Conclusion
Remarkably, we managed to achieve a model with a higher winning percentage than losses, and this success was primarily attributed to the use of Convolutional Neural Network (CNN).

# Evaluation Results
| Method                        | Success Rate    |Failure Rate     |
| ----------------------------- | --------------- | --------------- |
| Convolutional Neural Network (CNN) | 60.0%      | 40.0%           |
| Multi-Layer Perceptron (MLP)       | 54.0%      | 46.0%           |
| K-Nearest Neighbors (KNN)          | 41.0%      | 59.0%           |
