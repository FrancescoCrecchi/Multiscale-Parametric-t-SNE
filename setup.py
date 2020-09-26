
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="msp_tsne", # Replace with your own username
    version="0.0.1",
    author="Francesco Crecchi",
    author_email="francesco.crecchi@di.unipi.it",
    description="Keras implementation of Multiscale Parametric t-SNE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FrancescoCrecchi/Multiscale-Parametric-t-SNE",
    # packages=setuptools.find_packages(),
    packages=['msp_tsne'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'numba',
        'scikit-learn',
        'tqdm',
        'tensorflow==1.15.4',
        'keras==2.3.1'
    ]
)
