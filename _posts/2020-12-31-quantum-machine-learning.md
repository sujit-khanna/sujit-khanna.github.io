---
layout: post
title: Quantum Machine Learning - Assessing advantage over classical machine learning algorithms
---
In this article we take a look at how state-of-the-art quantum machine learning classifiers perform against a classical machine learning algorithm like multi-layer perceptron and check if it exhibits any form of quantum advantage over the classical algorithm.

Quantum algorithms are built specifically to operate on quantum computers which are fundamentally different from classical computers. The information in classical computers are stored in bits which take binary values of 0 or 1, whereas the quantum computer operates on qubits (quantum bits). Each qubit can assume a state of |0>, |1>, or in superposition of these two states a|0> + b|1>, 
where a,b are complex numbers. This enables qubits to take multiple values at a time which helps in performing computations which is not possible using a conventional computer. The qubits can also be entangled where multiple qubits are intertwined and measuring one qubit can tell is the state of the other and collapse the superposition irrespective of the distance at which these qubits are separated (can be light years away), the phenomenon of quantum teleporation and faster than light communication (proven to be impossible currently) are based on this concept.

## Quantum Machine Learning
A quantum algorithm is a set of instructions that is implemented on a quantum circuit to solve a given problem. A quantum circuit consists of a set of coherent quantum operations performed over quantum data using a single or multi-qubit gates. Grover's algorithm that can search an unsorted database in O(sqrt(N)), and Shor's algorithm that can factor integers in polynomial time are one of the most well known quantum algorithms. The main motivation behind quantum ML algorithms is based on the fact that ```if small quantum information processors can produce statistical patterns that are computationally difficult to be producted by classical computers, then perhaps they can also recognize patterns that are equally difficult to recognize classically[1]```.  . These Quantum ML algorithms can potentially reduce training samples as well, meaning they would require less data to learn patterns, thus exhibiting an advantage over classical ML algorithms.

### Varational Circuits
Quantum ML algorithms are generally represented by varational circuits which are also known as parameterized quantum circuits. As these circuits contain free parameters and are differentiable they become ideal candidates to represent parameters of classical machine learning models. As these circuits are differentiable they can be optimized by classical iterative gradient based optimization techniques to find the best parameter set θ'. 
A varational circuit typically consists of 3 components,

* A fixed initial state which can be a zero state |x> or vacuum state
(for photonic quantum computer)
* A quantum circuit U(θ), 
that consists of a sequence of unitary gates that depends of the varational circuit parameter θ that are applied to circuit inputs |x>. This component prepares the final state which needs to be measured. The final state can be represented as U(x, θ)|0>.
* The final component measure the state prepared by the quantum circuit at the output of the varational circuit. (image taken from [9])

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/103393784-a1ade000-4af2-11eb-9526-2cabb5dd4a71.png"></p>


### Xanadu's Pennylane
All quantum ML algorithms described in this article are implemented using the pennylane library (insert link), which is built on python 3. This framework provides optimization routines for varational circuits and integrates the automatic differentiation algorithms (used in pytorch and tensorflow) into quantum hardware and simulators. Thus allowing one to create complex ML algorithms using only quantum layers as well as quantum and classical layers i.e. hybrid architectures. Another key benefit to using pennylane is that it is hardware agnostic, i.e. we can access any quantum hardware as a backend (QISKIT, Rigetti, Xanadu, etc.) using a custom plugin. 

## Quantum ML Classifiers
In the following sections we will primarily implement two types of quantum classifiers, which are quantum neural network and Classical-Quantum (Hybrid) Neural Network, and evaluate its performance against a standard 2 hidden layered multi-layer perceptron. The dataset used in these evaluations is the IRIS dataset which is a multiclass classification problem. Since majority of tutorials online focus on binary classification performance of quantum ML algorithms, multiclass problem could shed some interesting insights. There are a couple of tutorials available that describe a quantum multi-class classifiers that use one vs all classifiaction technique which basically consists of 3 different models trained to classify 3 different classes of the IRIS dataset, however our implementation uses a single model with appropriate number of qubits to classify multi-class labels of the IRIS dataset.
### Quantum Neural Networks (QNN)
QNN or varational networks only uses varational circuits in its architecture, i.e. no classical components are used in its implementation. To use such a network for real-world applications the input data must first be encoded into a quantum form that can be used by a quantum circuit. This step is called ***State Preparation***, which encodes the real vectors as amplitue vectors, this is a necessary step as real valued input features are not differentiable when directly passed on to the varational circuit. The ***State Preparation*** can be done in pennylane using the ***AmplitudeEmbedding*** function, whch encodes the 2^n input features into amplitudes of number of wires in the circuit or n qubits. This function first normalizes the input features and pads the excess features in case of num_features <= 2^n - 1 number of qubits. Once the input features are embedded as amplitudes of qubits, the next set of operations can involve application of any number of arbitrary gates. 

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/103395691-6c5abf80-4afd-11eb-811d-3075937c1a41.png"></p>



In our implementation we followed the amplitude embedding with a layer of ***StronglyEntanglingLayers*** as the next ste of quantum operations. The ***StronglyEntanglingLayers*** gates consists of a single qubit rotations followed by entangling of qubits using the CNOT gates. The two components i.e. ***AmplitudeEmbedding*** and ***StronglyEntanglingLayers*** form a single block of variational circuit that will be used to represent a single hidden layer of a classical neural network. In our implementation we will repeat this block 3 times to represent 3 hidden quantum layers. The figure below represents the relation between classical neural network and a quantum neural network.

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/103395725-97ddaa00-4afd-11eb-8d73-369a6ab2eb03.png"></p>


### Classical-Quantum (Hybrid) Neural Network
A Classical-Quantum (Hybrid) Neural Network is a type of a neural network that contains both quantum layers (varaitional circuits) as well as classical layers. In our implementation we stack quantum layers in-between two classical hidden neural network layers. The quantum layers are implemented in exactly the same manner as discussed in the last subsection and classical hidden layers are fully connected neural layers followed by activation units (’tanh’ for the first hidden layer and ’softmax’ for the last hidden layer). As variational circuits are differentiable similar to classical neural networks, such a hybrid architecture can be trained as a whole with existing iterative gradient based methods like SGD, ADAM, RMSProp, etc. The figure below represents the architecture of such a hybrid network.

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/103395791-f2770600-4afd-11eb-8363-6cd86e52ec6a.png"></p>

In our implementatio since we're using two classical layers sandwiching a quantum layer built on 4 qubits with 3 varational circuit blocks, the actual architecture of the hybrid model used is shown below. Here the oututs represent the outputs of a softmax layer.

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/103395839-26522b80-4afe-11eb-9795-86128bb78bdf.png"></p>

## Experimental Results
We use the IRIS dataset used in this analysis, which as 4 input features and 3 output classes, with 150 records (rows). For each model we split the training and test set with 75%:25% split, and plot necessary loss and accuracy relations along with classification metrics. Since this is a multiclass classification problem the loss function minimized is categorical cross entropy.

### Quantum Neural Network
The quantum neural network implemented for this experiment has 3 hidden layers i.e. 3 variational circuit blocks, and implemented with 3 qubits, the optimization routine used in ***ADAM*** with learning rate set to L_R = 0.1, the model is trained for atmost ***100*** epochs. The results can be seen in plots and tables below. This model achieves an overall accuracy of ***92%*** on the test set.

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/103396150-7c739e80-4aff-11eb-8f33-f0cb09e223c8.png"></p>
   Figure 6: Quantum Neural Network Categorical cross entropy loss, training accuracy and test accuracy vs num_epochs


<p align="center"><img src="https://user-images.githubusercontent.com/71300644/103396189-b80e6880-4aff-11eb-8908-8817abdafbb7.png"></p>

### Hybrid Neural Network

The hybrid neural network implemented for this experiment has 3 variational blocks i.e. layers slotted in-between 2 classical layers.The activation function used in the first classical hidden layer is ***tanh*** and in the final hidden layer is ***softmax***. A total of 4 qubits are used for this model, ***ADAM*** optimizer with learning rate set to L_R = 0.1 is used for training the model. This model is trained for ***20*** epochs, and achieves a test set accuracy of ***100%***. The detailed results can be seen in plots and tables below.

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/103396251-1b989600-4b00-11eb-92da-be27b411b952.png"></p>
Figure 8: Hybrid Neural Network Categorical cross entropy loss, training accuracy and test accu- racy vs num_epochs

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/103396268-3b2fbe80-4b00-11eb-80e8-a1bc18a7eb13.png"></p>

### Classical Neural Network (Multi_layer Preceptron)
We use a fairly simple classical neural network, also know as a multi-layered preceptron. A total of 3 hidden layers are used in this model, with activation units being ***tanh*** for all hidden layers except the final layer. The final layer has ***softmax*** as its activation unit for the classifying the input features.ADAM optimizer with learning rate set to L_R = 0.1 is used for training the model for ***20*** epochs, and the trained achieves a test set accuracy of ***92%***.The detailed results can be seen in plots and tables below.

<p align="center"><img src="https://user-images.githubusercontent.com/71300644/103396322-877afe80-4b00-11eb-8a63-60bf366de946.png"></p>
Figure 10: Classical Neural Network Categorical cross entropy loss, training accuracy and test accuracy vs num_epochs


<p align="center"><img src="https://user-images.githubusercontent.com/71300644/103396359-c9a44000-4b00-11eb-9122-ca996e6a1c91.png"></p>

### Comparative Analysis
The classical neural network used in the experiments serves as the benchmark against which we compare our quantum machine learning algorithms. We use a host of classification metrics like precision, recall, f-1 score, accuracy and jaccard similarity to assess the performance of each model. The table below compares all three neural networks based on these classification metrics. Based on the table above we can observe that Hybrid Neural network tends to marginally outperform both Quantum neural networks and classical neural networks. The Quantum neural networks performs as well as the classical neural networks on all classification metrics except jaccard similarity where it marginally underperforms the classical neural network.


<p align="center"><img src="https://user-images.githubusercontent.com/71300644/103396377-f0627680-4b00-11eb-9d80-feb53b96cf22.png"></p>


## Conclusion
In this project we implemented two types of quantum neural networks and bench-marked them against a classical neural network. From the comparative analysis we could observe that neither Quantum nor the Quantum-Classical neural network clearly exhibited a quantum advantage over the classical one, however they performed better or atleast as good as classical neural networks on the IRIS dataset. This gives us some encouragement as there could be datasets or applications where quantum neural networks (including hybrid neural networks) may exhibit some form of quantum advantage. Currently due to hardware limitations deeper and wider circuits cannot be tested exhaustively, however over the next few years this will change and allow development of complicated and larger/deeper machine learning algorithms. We also foresee a few challenges and bottlenecks with quantum machine learning algorithms in NISQ (Noisy Intermediate-Scale Quantum) era enumerated below.
*   ***Benchmarking***: It is diﬀicult to determine whether an algorithm provides a quantum ad- vantage over a classical algorithm, as machine learning algorithms can easily overfit on the training dataset.
*   ***Input data loading***: If this component dominates the running time, we cannot get a quantum advantage.
*   ***Intractability***: The number of gates which could provide a quantum speedup, even if it exists, is unknown.


The repository containing the relevant notebooks can be accessed here 
[Quantum Machine Learning](https://github.com/sujit-khanna/Quantum_Machine_Learning)

## References ##
    1.  Jacob Biamonte, Peter Wittek, Nicola Pancotti, Patrick Rebentrost Nathan Wiebe, and Seth Lloyd. Quantum machine learning. Nature, 549(7671):195–202, Sep 2017. 
    2. Pennylane community. pennylane, 2020.
    3. IRIS DATASET http://archive.ics.uci.edu/ml/datasets/Iris/. Uci repository.
    4. S Ahmed (https://pennylane.ai/qml/demos/tutorial_data_uploading_classifier.html). Data Uploading Classifier. pennylane.
    5. pennylane (https://pennylane.ai/qml/demos/tutorial_qnn_module_tf.html). Turning quantum nodes into keras layers. 
    6. pennylane Tutorials (https://pennylane.ai/qml/demos/tutorial_multiclass_classification.html). Multiclass margin classifier. pennylane, 2020. 
    7. Xanadu (pennylane.ai). pennylane.ai.
    8. Pennylane Tutorials(https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html). Quantum transfer learning. pennylane, 2020. 
    9. Pennylane Varational Circuits (https://pennylane.ai/qml/glossary/variational_circuit.html)

