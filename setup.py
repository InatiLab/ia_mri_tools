#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=6.0',
    'numpy>=1.10',
    'scipy>=0.18'
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='ia_mri_tools',
    version='0.1.0',
    description="Python tools for MRI data analysis.",
    long_description=readme + '\n\n' + history,
    author="Souheil James Inati",
    author_email='souheil@inatianalytics.com',
    url='https://github.com/inati/ia_mri_tools',
    packages=[
        'ia_mri_tools',
    ],
    package_dir={'ia_mri_tools':
                 'ia_mri_tools'},
    entry_points={
        'console_scripts': [
            'ia_mri_tools=ia_mri_tools.cli:main'
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    license="BSD license",
    zip_safe=False,
    keywords='ia_mri_tools',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
