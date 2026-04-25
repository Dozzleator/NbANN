# NbANN

## Description
NbANN is a lightweight, custom-built Multi-Layer Perceptron (MLP) neural network library. Developed entirely from scratch using only Python and NumPy. The library features a clean, Scikit-Learn style API (.fit(), .predict(), .accuracy()), making it an excellent tool for deep learning education, algorithmic transparency, and rapid prototyping.

## Dependencies
This project is built to rely on minimal external packages.
* Python 
* numpy
* setuptools 

## Setup and Installation
To install the package locally in editable mode (which is recommended for development and testing), navigate to the root directory where the `pyproject.toml` file is located and run the following command in your terminal:

```bash
pip install -e .
```

Else for installation through pip run following pip command in python terminal:

```bash
pip install NbANN
```

## File Structure
The project is cleanly modularised into specific mathematical components to ensure readability and easy maintenance:

```bash
nba_project/
│
├── pyproject.toml       
└── nbann/
    ├── __init__.py          
    ├── core.py                 
    └── components/
        ├── __init__.py
        ├── build_layers.py     
        ├── foward_pass.py      
        ├── loss_functions.py   
        ├── back_propagation.py 
        └── optimiser.py       
```

## Author
- Name: Joseph Whakaari
- Email: jwhakaari@gmail.com