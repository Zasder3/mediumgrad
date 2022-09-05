import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mediumgrad",
    version="0.1.0",
    author="Cade Gordon",
    author_email="cadegord@berkeley.edu",
    description="A tiny vector-valued autograd engine with a small PyTorch-like neural network library on top.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Zasder3/mediumgrad",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
