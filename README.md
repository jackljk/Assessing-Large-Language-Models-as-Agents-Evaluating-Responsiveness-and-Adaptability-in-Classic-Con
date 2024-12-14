# Assessing LLM as Agents for Classic Control Tasks
This repository contains the code for the paper "Assessing LLM as Agents for Classic Control Tasks" for DSC190 ML with Few Labels by Jack Kai Lim, Mohit and Andrew.

## Bipedal Walker
The code and scripts for the Bipedal Walker environment can be found in the `Bipedal Walker` directory. The `Bipedal Walker` directory contains the following directories:
- `notebooks`: Contains the Jupyter notebooks used for testing
- `recording`: Contains the videos of the agents playing the Bipedal Walker environment
- `runs`: Contains the files for the observations saved during training
- `src`: Contains the source code for the Bipedal Walker environment

### How to run the Bipedal Walker environment
Install required packages:
```bash
pip install -r Bipedal\ Walker/requirements.txt
```

Run the training script:
```bash
python Bipedal\ Walker/src/main.py
```
To adjust the hyperparameters and play around with the different experiments, you can modify the `Bipedal Walker/src/default.yaml` file which uses `hydra` to manage the configurations.

The training also currently uses OpenAI API for the LLM model, which requires to have a valid API key. You can get the API key from the [OpenAI website](https://beta.openai.com/account/api-keys). Then store the API key in a `.env` file in the root directory of the project with the key as `OPENAI_API_KEY`.

## Lunar Lander


## Car Racing