from setuptools import setup, find_packages

setup(
    name='BORL',
    version='0.0.1',
    python_requires='>=3.5.0',
    keywords='Bayesian Optimization, RL',
    packages=[package for package in find_packages()],
    install_requires=[
        'gym>=0.9.6',
        'numpy',
        'pyqt5>=5.10.1',
		'scipy',
		'ipdb',
		'matplotlib'
    ],
    package_data={}
)