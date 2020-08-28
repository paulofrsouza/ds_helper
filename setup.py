from setuptools import find_packages, setup

with open("README.md", "r") as fh:
	long_description = fh.read()

setup(
    name='ds_helper-paulofrsouza',
    packages=find_packages(),
    version='0.1.0',
    description='Collection of helper functions for routine Data Science tasks.',
	 long_description=long_description,
	 long_description_content_type="text/markdown",
    author='Paulo Souza',
	 url="https://github.com/paulofrsouza/ds_helper",
    license='MIT',
	 classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Ubuntu Linux 18.04+",
    ],
	 python_requires='>=3.6',
)
