# fake_news_project
Our project consists of an analysis comparing probabilistic models, such as logistic regression, to deep learning models, such as recurrent neural networks in howe well they can detect fake vs real news.

You can observe both the code for the probabilistic models and the deep learning one usign pytorch and different dependencies. 



## Installation

Dependencies are handled with conda, you can install them with:
```s
conda env create -f environment.yml
```
Then activate the environment with
```sh
conda activate ml_project
```
The project can then be installed with pip:
```sh
pip install .
```

## Tests

You can run the tests with pytest by running the following command in the root directory of the project:
```sh
pytest -v
```
app.py(older version of prototype.py) was used for pytest. 

## Usage
Our project can be used through 
