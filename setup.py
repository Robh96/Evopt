from setuptools import setup, find_packages

setup(
    name='evopt',
    version='0.136',
    packages=find_packages(),
    install_requires=[
        'numpy>=2.2.1',
        'pandas>=2.2.3',
        'cma>=4.0.0',
        'cloudpickle>=3.1.1',
        'scipy>=1.15.0',
        'plotly>=5.24.1',
        'matplotlib>=3.10.0',
    ],
    include_package_data=True,
    author='Roberto Hart-Villamil',
    author_email='rob.hartvillamil@gmail.com',
    description='User Friendly Black-Box Numerical Optimization Package in Python.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Robh96/Evopt',
    license='GNU',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.13.1',
    keywords=(
        'optimization evolutionary cmaes calibration'
        'simulation fine-tuning simple'
    ),
)
