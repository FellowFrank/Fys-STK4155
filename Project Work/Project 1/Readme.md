READ ME 
--------------------------------------------
*Group Members*: Anton Nicolay Torgersen

## Description
Delivered as a first project to the get a grade in the subject FYS-4155 at the University of Oslo autumn 2025. This is project 1 of 3, where the task in this project is to create a scientific report where the student goes through the different regression methods on Runge's function. Then looking into some resampling techniques on the OLS method.
The Method i wanted to answer the project in was to look at a sparse graph and focus on how the different methods would respond to having few training points.


### Code
To run the codes used in this report, simply create a virtual environment and install the packages in ``Code/requirements.txt`` with pip or an other package manager. 
For pip and using windows this would be the following:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

#### ``Code/Utils.py``
A python library for functions defined in the project for easy of use.
Includes the following:
- Generation of dataset
- the Polynomial desing matrix
- OLS methods(Analytical and Gradient)
- Ridge methods(Analytical and Gradient)
- Lasso gradient method using soft thresholding method

### ``Code/ResultsGeneration.ipynb``
The main code for the research and experiments done in this project.
It is structured into sections just as the report is and contains the implementation of that method used in the code.
It Contains:
- A comparison of code in Utils and Scikit
- OLS Analysis
- Ridge Analysis
- Lasso Analysis
- Gradient descent analysis with varing learning rates
    - Standard
    - Momentum
    - Adagrad
    - RMSprop
    - ADAM
- Stochastic Gradient descent
- Two sampling methods
    - bootstrap
    - k-fold CV