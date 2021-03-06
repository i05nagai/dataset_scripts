#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup
from setuptools.command.test import test as TestCommand
import sys


NAME = "misc"
MAINTAINER = "i05nagai"
MAINTAINER_EMAIL = ""
DESCRIPTION = """ """
with open("README.rst") as f:
    LONG_DESCRIPTION = f.read()
LICENSE = ""
URL = ""
VERSION = "0.0.1"
DOWNLOAD_URL = ""
CLASSIFIERS = """ \
Development Status :: 1 - Planning
Intended Audience :: Science/Research
Intended Audience :: Developers
Programming Language :: Python
Programming Language :: Python :: 3.5
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Unix
Operating System :: MacOS
"""


class PyTest(TestCommand):
    """
    """
    user_options = [
        ('cov=', '-', "coverage target."),
        ('pdb', '-', "start the interactive Python debugger on errors."),
        ('pudb', '-', "start the PuDB debugger on errors."),
        ('quiet', 'q', "decrease verbosity."),
        ('verbose', 'v', "increase verbosity."),
        # collection:
        ('doctest-modules', '-', "run doctests in all .py modules"),
    ]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.cov = 'misc'
        self.pdb = False
        self.pudb = False
        self.quiet = False
        self.verbose = True
        self.pep8 = False
        self.doctest_modules = False
        self.bench = False

        self.test_args = ["misc"]
        self.test_suite = True

    def finalize_options(self):
        TestCommand.finalize_options(self)

    def run_tests(self):
        import pytest
        # if cov option is specified, option is replaced.
        if self.cov:
            self.test_args += ["--cov={0}".format(self.cov)]
        if self.pdb:
            self.test_args += ["--pdb"]
        if self.pudb:
            self.test_args += ["--pudb"]
        if self.quiet:
            self.test_args += ["--quiet"]
        if self.pep8:
            self.test_args += ["--pep8"]
        if self.verbose:
            self.test_args += ["--verbose"]
        if self.doctest_modules:
            self.test_args += ["--doctest-modules"]

        print("executing 'pytest {0}'".format(" ".join(self.test_args)))
        errno = pytest.main(self.test_args)
        sys.exit(errno)


def main():
    cmdclass = {
        'test': PyTest,
    }
    metadata = dict(
        name=NAME,
        packages=[NAME],
        version=VERSION,
        description=DESCRIPTION,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        url=URL,
        download_url=DOWNLOAD_URL,
        license=LICENSE,
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        long_description=LONG_DESCRIPTION,
        tests_require=['pytest', 'pytest-cov'],
        cmdclass=cmdclass
    )

    setup(**metadata)


if __name__ == '__main__':
    main()
