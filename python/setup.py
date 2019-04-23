from setuptools import setup, find_packages

setup(
    name='BORL_python',
    version='0.0.1',
    python_requires='>=3.5.0',
    keywords='Bayesian Optimization, RL',
    packages=[package for package in find_packages()
                if package.startswith('BORL_python')],
    install_requires=[
        'gym>=0.9.6',
        'pyqt5>=5.10.1',
		'ipdb',
		'matplotlib'
        'tqdm'
        'pickle'
        'dill'
    ],
    package_data={}
)
