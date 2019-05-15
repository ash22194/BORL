# BORL
Ongoing work on using Bayesian Optimization for Policy Learning

## Python
The code has been tested with Python versions 3.6.4 and 3.7
### Installing Dependencies
- The `setup.py` script located in `<path-to-BORL>/python/` specifies the required packages to run the code. Run the following command to setup the dependencies and install the `BORL_python` package. You may also create and activate a virtual environment before running this script
`pip install -e <path-to-BORL>/python/`

- Install [climin](https://github.com/BRML/climin) package from the GitHub repository.
	````
	git clone https://github.com/BRML/climin.git <path-to-climin>
	cd <path-to-climin>
	pip install -e .
	````

### Test Code
The algorithms GPTD, GPSARSA and their variations are implemented as different classes and can be found in the `<path-to-BORL>/python/BORL_python/value` folder. 

#### Before running the test scripts
Some of the algorithms reuse previously learned value functions and policies in building a policy. **Make sure** to create the directories `<path-to-BORL>/python/BORL_python/data/GPSARSA` and `<path-to-BORL>/python/BORL_python/data/GPTD` **before** running the test scripts. If executing for the first time, the test scripts would create the required value functions and policies before running the TD/SARSA algorithms and store them in these directories. Moreover, the test scripts use these directories for logging other relevant data. 

#### Running the test scripts
Test scripts for running these algorithms can be found in the folder `<path-to-BORL>/python/BORL_python/test`. Currently, we have only tested these algorithms on the pendulum swing-up problem and therefore the test scripts follow the naming convention `swingUp<algorithm-name>.py` 

## MATLAB
Instructions to run scripts would be updated soon.
