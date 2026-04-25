# NbANN

## Description
NbANN is a lightweight, custom-built Multi-Layer Perceptron (MLP) neural network library. Developed entirely from scratch using only Python and NumPy, it strips away the "black-box" nature of heavy frameworks like PyTorch or TensorFlow. Originally inspired by NBA statistical forecasting, the library features a clean, Scikit-Learn style API (.fit(), .predict()), making it an excellent tool for deep learning education, algorithmic transparency, and rapid prototyping.

## Dependencies
This project is built with a "Zero Magic" philosophy, relying on minimal external packages.
* Python 3.x
* numpy >= 1.20.0
* setuptools >= 61.0 (for the build system)

## Setup and Installation
This project uses a modern `pyproject.toml` build system. To install the package locally in editable mode (which is recommended for development and testing), navigate to the root directory where the `pyproject.toml` file is located and run the following command in your terminal:

```bash
pip install -e .
```

## File Structure
The project is cleanly modularised into specific mathematical components to ensure readability and easy maintenance:

```bash
nba_project/
│
├── pyproject.toml       
└── mlp/
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

## Extra
Although the code is open source and available for free distribution it has also been uploaded to PyPI for easy installation through pip.

```bash
pip install NbANN
```

## Author
- Name: Joseph Whakaari
- Email: jwhakaari@gmail.com