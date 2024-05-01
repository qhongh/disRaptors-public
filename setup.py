from setuptools import setup, find_packages

requirements = []
with open('requirements.txt', 'rt') as f:
    for req in f.read().splitlines():
            requirements.append(req)


setup(
    name="disraptors",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.10",
)