READ ME 
--------------------------------------------
*Group Members*: Anton Nicolay Torgersen

## Description
This is project 3 of 3, where the task in this project is to create a scientific report where the task was to analyze a dataset from https://www.kaggle.com/ using the some of the techniques learned through this course.
The dataset chosen for this project was the real waste dataset, https://archive.ics.uci.edu/dataset/908/realwaste, and was analyzed using these two methods FFNN and CNN.  



### Code
To run the codes used in this report, simply create a virtual environment and install the packages in ``Code/requirements.txt`` with pip or an other package manager. 
For pip and using windows this would be the following:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```



#### Code/Functions.py

Includes all the defined functions that we have utilized in this project:
- Dataset Retrieval
- Dataset Augmentation
- The Optimal CNN model found through the parameter search
- The Optimal FFNN model found through the parameter search on PCA 95%

#### Code/Workline.ipynb
The main execution pipeline (notebook) for the project. It details the step-by-step workflow used to obtain the results and final models utilized in the figures.ipynb.
- Data preprocessing and dimensionality reduction (PCA).
- Hyperparameter tuning experiments.
- Training loops for the FFNN and CNN models.

#### Code/Figures.ipynb
Generates all the figures used in the text.
It is created as a readable file where one can see the different training runs for the models and the extra figures not utilized, like the confusion diagrams.  