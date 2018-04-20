from setuptools import setup, find_packages
from setuptools.command.install import install
import re
import os
import pip

try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements

with open(os.path.join(os.path.dirname(__file__), 'optimization', 'version.py')) as v_file:
    VERSION = re.compile(r".*__version__ = '(.*?)'", re.S).match(v_file.read()).group(1)


class OverrideInstall(install):
    """
    Emulate sequential install of pip install -r requirements.txt
    """

    def run(self):
        # install requirements first
        install_reqs = parse_requirements(
            './requirements/common.txt', session=False)
        reqs = [str(ir.req) for ir in install_reqs]
        for req in reqs:
            pip.main(["install", req])

        install.run(self)


config = {
    'name': 'optimization',
    'description': 'Package for mathematical optimization',
    'author': 'G. Donetti',
    'url': '',
    'download_url': '',
    'author_email': '',
    'version': VERSION,
    'packages': find_packages(),
    'scripts': [],
    'cmdclass': {'install': OverrideInstall},
    'test_suite': 'nose.collector',
    'tests_require': ['nose'],
}

setup(**config)
