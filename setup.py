from setuptools import setup, find_packages

__version__ = '0.1'
url = ''

install_requires = ['torch',
                    'numpy',
                    'matplotlib',
                    'nibabel',
                    'scipy',
                    'PyQt5,
                    'Pillow-PIL',
                    'pydicom']

setup(
    name='GReAT4Torch',
    description='Groupwise Registration Algorithms and Tools for Torch',
    version=__version__,
    author='Kai Brehmer',
    author_email='k.brehmer@uni-luebeck.de',
    url=url,
    keywords=['image registration, groupwise image registration, groupwise, image computing'],
    install_requires=install_requires,
    packages=find_packages(exclude=['build']),
    ext_package='')
