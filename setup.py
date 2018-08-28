#!/usr/bin/env python


from setuptools import setup


setup(name='odoo_transfer_lib',
      version='0.0.1',
      description='Simple library to transfer data betwen Odoo instances',
      author='Center of Research & Development',
      # author_email='info@crnd.pro',
      # url='https://crnd.pro',
      py_modules=['odoo_transfer_lib'],
      license="LGPL",
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
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
      keywords=['odoo', 'data-transfer', 'odoo-rpc', 'rpc', 'xmlrpc',
                'xml-rpc', 'json-rpc', 'jsonrpc', 'odoo-client', 'openerp'],
      install_requires=[
          'odoo_rpc_client',
      ],
)
