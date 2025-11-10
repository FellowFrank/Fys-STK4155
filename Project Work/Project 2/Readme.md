READ ME 
--------------------------------------------
*Group Members*: Anton Nicolay Torgersen

## Description
Delivered as a second project to the get a grade in the subject FYS-4155 at the University of Oslo autumn 2025. This is project 2 of 3, where the task in this project is to create a scientific report where the student writes there one neural network code and compares the results after training with what was found in project 1.
The second part is using the same methods developed in part 1 in doing a classification analysis of the MNIST problem classifying the different handwritten images of numbers to the correct number.



### Code
To run the codes used in this report, simply create a virtual environment and install the packages in ``Code/requirements.txt`` with pip or an other package manager. 
For pip and using windows this would be the following:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```


#### Code/Functions.py

Includes all the defined functions that we have created in this project separated into different sections:
- Dataset Generation for Regression
- Activation Functions & derivatives
- Loss Functions & derivatives
    - Cross_entropy uses der_softmax as its derivative, see methods in the paper.
- Optimization Algorithms ( Adam and RMSprop)
    - Gradient descent and Stochastic Gradient descent, is integrated into the Neural Network class and the run experiment function.
- A Neural Network class with:
    - Initiation of layers
    - Predictions & Cost
    - Gradients
    - Update of layers
    - Plus 2 tests to see if it's configured correctly
- Two broad functions to run an experiment and to graph the heath map
    - was created to reduce code

#### Code/Verification.ipynb
A document that verifies that the neural network has been correctly implemented.
By calculating the gradients and doing a test run on the dataset.

#### Code/GridSearchParameters.ipynb
A document that does the three grid search written about in the text, are very computationally expensive.
Was used to get a baseline for further tuning or analysis.

#### Code/Workline.ipynb
Not important.
Is a notebook on how the project was executed and the different steps in order to answer the parts and components for this project.

#### Figures/Figures.ipynb
Generates all the figures used in the text. 