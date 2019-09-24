# recommender101
Playing around with blog posting and recommender systems

# How to run the thing
## How to run
This project is based on python 3.7 and uses features like dataclass that is only available since 3.7. It
uses poetry for dependency management. 
So, when you do have a python 3.7 installation, the steps to get your environment up and running are
```bash
# Install poetry through pip. You will find poetry in ~/.poetry/bin and can add it to your .bashrc as 
# export PATH=$HOME/.poetry/bin:$PATH
pip install poetry
# With this, the virutal environemnts created by poetry are located within the project. Helpful for ides like pycharm
# and vs code
poetry config settings.virtualenvs.in-project true
# Change to the directory where you cloned this project into
cd recommender101 # The folder that you've clone the repo into
# Install all dependencies. This might take a time as it creates the virutal environments 
poetry install
```

