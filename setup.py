#!/usr/bin/env python

# Copyright © 2016-2019 Center of Research & Development <info@crnd.pro>

#######################################################################
# This Source Code Form is subject to the terms of the Mozilla Public #
# License, v. 2.0. If a copy of the MPL was not distributed with this #
# file, You can obtain one at http://mozilla.org/MPL/2.0/.            #
#######################################################################

from setuptools import setup


setup(
    name='odoo_transfer_lib',
    version='0.0.1',
    description='Simple library to transfer data betwen Odoo instances',
    author='Center of Research & Development',
    author_email='info@crnd.pro',
    url='https://crnd.pro',
    py_modules=['odoo_transfer_lib'],
    license="MPL 2.0",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Utilities',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords=[
        'odoo', 'data-transfer', 'odoo-rpc', 'rpc', 'xmlrpc',
        'xml-rpc', 'json-rpc', 'jsonrpc', 'openerp'],
    install_requires=[
        'odoo_rpc_client',
    ],
)
