from setuptools import setup

with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

setup(name='gym_pybullet_drones',
    version='0.6.0',
    packages=['logs', 'wandb', 'files', 'script', 'experiments', 'gym_pybullet_drones'],
    install_requires=required_packages,
)