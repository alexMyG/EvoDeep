# EvoDeep
### A new Evolutionary approach for automatic Deep Neural Networks parametrisation

This repository contains the necessary code to run EvoDeep, an evolutionary approach to adjust Deep Neural Network parameters

##### Authors

Alejandro Martín, Raúl Lara-Cabrera, Félix Fuentes-Hurtado, Valery Naranjo, David Camacho

##### Documentation
For a full description of EvoDeep, please read the following publication:

> Martín, Alejandro, et al. "EvoDeep: A new evolutionary approach for automatic Deep Neural Networks parametrisation." Journal of Parallel and Distributed Computing 117 (2018): 180-191.

##### Requirements
- EvoDeep has a series of dependencies. We recommend you to use a Virtual Environment:
    Install `virtualenv`:
    ```sh
    $ sudo pip install virtualenv
    ```
    Create virtual environment and activate it:
    ```sh
    $ virtualenv env
    $ source env/bin/activate
    ```
- The following Python libraries are required:
    ```sh
    $ pip install -r requirements.txt
    ```
    
By default, EvoDeep downloads the MNIST dataset and uses to train the different models, but you can set your own data in run.py

### Run EvoDeep
- You need to launch run.py in order to run EvoDeep:
    ```sh
    $ python run.py parametersGenetic.json --mu <mu> --lambda <lambda> --cxpb <cxpb> --mutpb <mutpb> --newpb <newpb> --ngen <ngen> --parallel
    ```
The previous call allows the following parameters:
* `-h`: Show the help message and exit.
* `--mu MU`: mu parameter of the genetic search
* `--lambda LAMBDA`: lambda parameter of the genetic search
* `--cxpb CXPB`: crossover probability
* `--mutpb MUTPB`: mutation probability
* `--newpb NEWPB`: probability of adding a new layer
* `--ngen NGEN`: number of generations
* `--seed SEED`: if not included, a default value is taken
* `--test-size TEST_SIZE`: ratio of observation for testing
* `--parallel`: if running evaluations in parallel
* `PARAMETERS_FILE`: a file defining layers and the range of values for each parameter. It also includes the finite state machine states definition (in charge of defining valid transitions between layers). Experiments were performed using **parametersGenetic.json**

- Execution example
    ```sh
    $ python run.py parametersGenetic.json --ngen 20 --mu 5 --lambda 10 --newpb 0.5 --cxpb 0.5 --mutpb 0.5 --parallel
    ```
