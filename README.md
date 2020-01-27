# odoo-transfer-lib

Simple library to transfer data bewen two Odoo instances.
Supported Odoo versions same as for [odoo-rpc-client](https://github.com/katyukha/odoo-rpc-client)
that are used underthe hood.

At this moment this library tightly integrated with [Jupyter Notebook](http://jupyter.org/)
and could be used only inside notebook.

## Notes

Currently it is required to run data-transfer from [Jupyter Notebook](http://jupyter.org/).
This is required to show progressbars. And it is not possible to run withour progress bars yet.

It is recommended to use [openerp-proxy](https://github.com/katyukha/openerp-proxy) - it provides jupyter integration for *odoo-rpc-client*

See [examples](./examples/) directory for more details.


## Example product transfer configuration


First we have to declare transfer configuration

```python
from odoo_ransfer_lib import TransferModel


class TProductCategory(TransferModel):
    model = 'product.category'
    transfer_fields = ['name', 'type', 'parent_id']
    auto_populate_cache = True

    def get_search_domain(self, category):
        domain = [('name', '=', category.name)]
        level = 1
        parent = category.parent_id
        while parent:
            parent_field = 'parent_id.' * level + 'name'
            domain += [(parent_field, '=', parent.name)]
            parent = parent.parent_id
            level += 1
        return domain


class TProductUOMModel(TransferModel):
    model = 'product.uom'
    auto_populate_cache = True
    link_field = 'name'


class TProductModel(TransferModel):
    model = 'product.product'
    transfer_fields = [
        'state',
        'uom_po_id',
        'uom_id',
        'product_code',
        'active',
        'purchase_ok',
        'categ_id',
        'description',
        'name',
        'description_sale',
        'volume',
        'type',
        'sale_ok',
        'default_code',
        'weight',
        'valuation',
    ]
    link_field = 'default_code'

    renamed_fields = {
        'hr_expense_ok': 'can_be_expensed',
    }
    auto_populate_cache = True
    populate_cache_domain = [('default_code', '!=', False)]

    # auto_transfer_fields
    auto_transfer_enabled = True
    auto_transfer_domain = []
    auto_transfer_priority = 10
    # ---

    def prepare_to_transfer(self, product):
        super(TProductModel, self).prepare_to_transfer(product)
        if not product.default_code:
            default_code = "auto-code-%s" % product.id
            product.write({'default_code': default_code})
            # Update cache of product record
            product._data['default_code'] = default_code
```

Next we can run data transfer

```python
# Import Client and Session classes
from odoo_rpc_client import Client, Session

# Connect to both databases
cl_from = Client(host='localhost', port='10069', dbname='test-data-transfer', user='admin', pwd='admin')
cl_to = Client(host='localhost', port='11169', dbname='test-data-transfer', user='admin', pwd='admin')

# Ensure connected
assert cl_to.uid
assert cl_from.uid

# Run transfer
transfer = Transfer(cl_from, cl_to, simplified_checks=True)
transfer.auto_transfer()

# Print transfer statistics
transfer.stat
```

## Launch your own ITSM system in 60 seconds

Create your own [Bureaucrat ITSM](https://yodoo.systems/saas/template/bureaucrat-itsm-demo-data-95) database

## Bug tracker

Bugs are tracked on [https://crnd.pro/requests](https://crnd.pro/requests>).
In case of trouble, please report there.

## Maintainer

![Center of Research & Development](https://crnd.pro/web/image/3699/300x140/crnd.png)

Our web site is: https://crnd.pro/

This module is maintained by the [Center of Research & Development](https://crnd.pro) company.

We can provide you further Odoo Support, Odoo implementation, Odoo customization, Odoo 3rd Party development and integration software, consulting services (more info available on [our site](https://crnd.pro/our-services)).Our main goal is to provide the best quality product for you. 

For any questions [contact us](mailto:info@crnd.pro>).

