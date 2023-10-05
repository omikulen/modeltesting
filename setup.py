from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='modeltesting',
    version='0.1',
    packages=find_packages(),    
    package_data={
        'modeltesting.neutrinos': ['data/*.npy'],
    },
    install_requires=requirements,
)