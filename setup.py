from setuptools import setup, find_packages

"""
Guidelines for setup.py:
https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#setup-args
"""
setup(
    name='scratch',
    version='0.1.2',
    description='Personal scratchpad for research, machine learning, and experimentation.',
    url='https://github.com/j93hahn/scratch',
    author='Joshua Ahn',
    author_email='jjahn@uchicago.edu',
    packages=find_packages(),
    python_requires='>=3',
    install_requires=[],
    entry_points={
        'console_scripts': [
            'circe=scratch.deploy.circe:main',
            'rune=scratch.deploy.rune_conda:main',
            'rune2=scratch.deploy.rune_apptainer:main',
        ]
    },
)
