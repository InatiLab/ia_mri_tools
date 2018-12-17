#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=6.0',
    'nibabel>=2.1',
    'numpy>=1.10',
    'scipy>=0.18'
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='ia_mri_tools',
    version='0.7.2',
    description="Python tools for MRI data analysis.",
    long_description=readme + '\n\n' + history,
    author="Souheil J. Inati",
    author_email='souheil@inatianalytics.com',
    url='https://github.com/inatiLab/ia_mri_tools',
    packages=[
        'ia_mri_tools',
    ],
    package_dir={'ia_mri_tools': 'ia_mri_tools'},
    entry_points={
        'console_scripts': [
            'ia_estimate_signal_mask=ia_mri_tools.cli:estimate_signal_mask',
            'ia_estimate_coil_correction=ia_mri_tools.cli:estimate_coil_correction',
            'ia_apply_coil_correction=ia_mri_tools.cli:apply_coil_correction',
            'ia_estimate_textures=ia_mri_tools.cli:estimate_textures',
            'ia_normalize_local=ia_mri_tools.cli:normalize_local',
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
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
