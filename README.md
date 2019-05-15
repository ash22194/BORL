
# BORL
Ongoing work on using Bayesian Optimization for Policy Learning

## Python
The code has been tested with Python versions 3.6.4 and 3.7
### Installing Dependencies
The `BORL_python/setup.py` script specifies the required packages to run the code. Run the following command to setup the dependencies and install the package. You may also create and activate a virtual environment before running this script

`pip install -e <path-to-BORL_python>/`

### Test Code
The algorithms GPTD, GPSARSA and their variations are implemented as different classes and can be found in the `BORL_python/value` folder. 

#### Before running the test scripts
Some of the algorithms reuse previously learned value functions and policies in building a policy. **Make sure** to create the directories `BORL_python/data/GPSARSA` and `BORL_python/data/GPTD` **before** running the test scripts. If executing for the first time, the test scripts would create the required value functions and policies before running the TD/SARSA algorithms and store them in these directories. Moreover, the test scripts use these directories for logging other relevant data. 

#### Running the test scripts
Test scripts for running these algorithms can be found in the folder `BORL_python/test`. Currently, we have only tested these algorithms on the pendulum swing-up problem and therefore the test scripts follow the naming convention `swingUp<algorithm-name>.py` 

## MATLAB
Instructions to run scripts would be updated soon.
