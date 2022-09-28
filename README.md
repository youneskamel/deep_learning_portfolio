# My Deep Learning portfolio 🧠
This is my deep learning portfolio. I will here share my various projects using deep neural networks, like research paper implementations. You will find below the descriptions of the various projects I worked on.

## Implementation of [Hasselt et al. (2015)](https://arxiv.org/abs/1509.06461) : Double Deep Q Learning on the Atari library with PyTorch :space_invader: 

In this project, the goal was to reproduce the solution offered by Hasselt et al. That solution improves on regular deep Q-learning by tackling one of its main weaknesses. Under regular deep Q learning, Q values are sometimes overestimated because of the way we compute the target values using the max function. The noise and random variation contained in estimated Q values means that some of them will be overestimated, and some of them underestimated. The fact that we use the max function on the Q values to choose the appropriate target value means that overestimates will tend to be chosen more often, leading to a Q value function that is biased towards larger estimates.

The solution proposed by the authors is to change the way target values are computed. Under regular deep Q learning, target values are computed by performing forward propagation with the target network using the new state, and then picking the maximum value among the values returned for each action. Under double deep Q learning, forward propagation is used on the new state but the action picked is not the one with the highest Q value as computed by the target network. Instead, we choose the action with the highest value on the new state computed by the online or predictor network. Since the two networks have different weights and thus different biases, by choosing the action with the highest Q value from the predictor network and feeding it as an argument to the target network, we avoid the problem of the self reinforcing bias caused by using the max function on the Q estimates of the target network.

In order to implement this solution, preprocessing the atari game frames was necessary. Since this type of preprocessing is specific to atari games and does not impact my understanding of deep Q learning, I decided not to invest a lot of time in that step. My preprocessing file utils.py thus draws heavily from Phil Tabor's solution from his Deep Q Learning course. The structure of the classes and functions of this repository are also inspired by Phil Tabor's solution for regular deep Q learning, as I thought that this organization was the most readable and understandable. The implementation of the ideas in the research paper, however, are my solutions.

## Implementation of [Wang et al. (2015)](https://arxiv.org/abs/1511.06581) : Dueling Double Deep Q Learning on the Atari library with PyTorch ⚔️

Here, the goal was to improve once more on the solution described above. Wang et al. suggest a new network architecture, splitting the output layer in two. The Q function that we try to estimate in Q-learning, in the action-state value. Given a certain state and a certain action, the Q-function will return the value of their combination, namely, using the given action in the given state. This Q function is thus the sum of the action function, A, and the state value function, V (Q = A + V). 
 
Wang et al. suggest splitting the last layer of the network in order to have two output layers, one that estimates A and the other V. This allowed for better performance because the dueling architecture can learn which states are valuable, without having to learn the effect of each action for each state. This is enables the network to identify states where its actions do not significantly impact the environment.
 
In order to implement this solution, preprocessing the atari game frames was necessary. Since this type of preprocessing is specific to atari games and does not impact my understanding of deep Q learning, I decided not to invest a lot of time in that step. My preprocessing file utils.py thus draws heavily from Phil Tabor's solution from his Deep Q Learning course. The structure of the classes and functions of this repository are also inspired by Phil Tabor's solution for regular deep Q learning, as I thought that this organization was the most readable and understandable. The implementation of the ideas in the research paper, however, are my solutions.


## Implementation of a convolutional autoencoder :elephant:

Description coming soon.
