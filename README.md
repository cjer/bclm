# bclm
THE go-to place for all Python Hebrew Treebank processing tasks.

## Installation

The installation is standard:

pip instatll <PATH_TO_WHEEL>

In order to create the wheel:
1. Make sure you have the latest versions of setuptools and wheel installed:
`python3 -m pip install --user --upgrade setuptools wheel`
2. Now run this command from the same directory where setup.py is located:
`python setup.py bdist_wheel`
3. It will generate a wheel file saved in the `dist` folder. 
4. You can now run `python3 pip instatll <PATH_TO_WHEEL>`
