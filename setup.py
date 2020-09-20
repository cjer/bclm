import setuptools
print(setuptools.find_packages())
with open("README.md", "r") as fh:
    long_description = fh.read()
    setuptools.setup(
        name='bclm',
        version='0.1',
        author="Dan Bareket",
        author_email="dbareket@gmail.com",
        description="THE go-to place for all Python Hebrew Treebank processing tasks.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/cjer/bclm",
        packages=['bclm'],
        install_requires=['pandas',
                          'conllu',
                          'numpy'],
        classifiers=["Programming Language :: Python :: 3",
                     "License :: OSI Approved :: MIT License",
                     "Operating System :: OS Independent", ],
    )
