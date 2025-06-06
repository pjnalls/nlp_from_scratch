<div align="center">
    <img 
        alt="metallic Earth with 'NLP' letters" 
        src="logo.avif"
        width="240px"
    />
</div>
<h1 align="center">
    Natural Language Processing<br/>
    (NLP) From Scratch
</h1>
<h3 align="center">
A walkthrough of an NLP character-level Recurrent Neural Network (RNN) and translation with a sequence-to-sequence network and attention provided by the PyTorch community that is available here: <a href="https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial" target="_blank"rel="noopener noreferrer">https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial</a>.
<hr>
</h3>

## Prerequisites
- Python 3 should be installed (I'm using Python 3.11 at the start of this project), which you can download here: https://www.python.org/downloads/
- A virtual environment created with a tool such as `venv` (I'm using it for this project), which you can learn how to set up here: https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/

## Setup Option A: Quickstart
If you have Python installed and are using a macOS or Linux machine, you can run the following script to create a `.venv` folder for your local virtual environment with the following command:
```bash
source setup.sh
```

## Setup Option B: Manual Project Setup
### via `venv`
Once you create a `venv` virtual environment with the guide, available [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/), inside a clone of this repo, run the following command:
```bash
pip install -r requirements.txt
```
### via `conda`
The `requirements.txt` file may be used to create an environment using the following command:
```bash
conda create --name your_env_name_here --file requirements.txt
```

## Running the Project
I'm using a Makefile to defined all of my CI commands at every step of the pipeline of the neural network model created for this project.

Here are some useful commands you can run at different stages of the pipeline:
### Download the data
```bash
make download
```
### Run the pipeline after the download step
```bash
make run
```
### Run the entire pipeline 
```bash
source make.sh
```
*Note: A Shell script is needed since new files are added to the system on the `make download` step and the process will only detect these changes during `make run` with a script like Shell not Make.*

### Delete compiled Python files
```bash
make clean
```
### Delete all data
```bash
make clean-data
```
### Delete all cached and external data
```bash
make clean-all
```
### ⚠️ NOTE FOR WINDOWS USERS ⚠️
Make is more challenging to install on Windows. I recommend using Chocolately guide here: https://earthly.dev/blog/makefiles-on-windows/