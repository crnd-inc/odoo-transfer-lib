import datetime
import collections

from odoo_rpc_client.plugins import external_ids
from odoo_rpc_client import Client
from odoo_rpc_client.orm.record import Record, RecordList
from odoo_rpc_client.orm.cache import empty_cache


class TransferError(Exception):
    pass


def in_progress(seq, msg="Progress: [%(processed)d / %(total)d]",
                length=None, close=True):
    """ Iterate over sequence, yielding item with progress widget displayed.
        This is useful if you need to precess sequence of items with some
        time consuming operations
        .. note::
            This works only in Jupyter Notebook
        .. note::
            This function requires *ipywidgets* package to be installed
        :param seq: sequence to iterate on.
        :param str msg: (optional) message template to display.
                        Following variables could be used in this template:
                            - processed
                            - total
                            - time_total
                            - time_per_item
        :param int length: (optional) if seq is generator, or it is not
                           possible to apply 'len(seq)' function to 'seq',
                           then this argument is required and it's value will
                           be used as total number of items in seq.
        Example example::
            import time
            for i in in_progress(range(10)):
                time.sleep(1)
    """
    from IPython.display import display
    from ipywidgets import IntProgress
    import time

    if length is None:
        length = len(seq)

    start_time = time.time()

    progress = IntProgress(
        value=0, min=0, max=length, description=msg % {
            'processed': 0,
            'total': length,
            'time_total': 0.0,
            'time_per_item': 0.0,
            'time_remaining': 0.0,
        }
    )
    display(progress)

    for i, item in enumerate(seq, 1):
        progress.value = i

        # i_start_time = time.time()

        yield item  # Do the job

        i_end_time = time.time()

        progress.description = msg % {
            'processed': i,
            'total': length,
            'time_total': i_end_time - start_time,
            'time_per_item': (i_end_time - start_time) / i,
            'time_remaining': ((i_end_time - start_time) / i) * (length - i),
        }

    if close:
        progress.close()


class GetOnlyOneID(object):
    """ Simple helper for data transfer.
        Try to get only single record by specified domain,
        and cache it for futher uses, or raise error
        if such record cannot be found, or there are more than
        one record found for specified domain
    """

    def __init__(self, cache_only=False):
        # if cache_only is set to True, then no search rpc calls
        # will be made to destination database, thus increasing speed of checks
        self.cache = collections.defaultdict(dict)
        self.cache_only = cache_only

    def get_model_cache(self, client:Client, model:str):
        """ Return model cache
        """
        return self.cache['c=%s|m=%s' % (client.get_url(), model)]

    def get_cache_key(self, domain:list):
        """ Converts domain to cache key
        """
        return tuple(domain)

    def __call__(self, client:Client, model:str, domain:list,
                 active_test=False, raise_on_gt_1=True, cached=True):
        """ Try to get only one record for specified domain.
            if there are more than one record found raise error (if raise_on_gt_1 is True)
            if no record found, just return False
            if cached is False, then no function internal cache will be used
        """
        if cached:
            mcache = self.get_model_cache(client, model)
            key = self.get_cache_key(domain)
            if key in mcache:
                return mcache[key]

        if self.cache_only:
            # if we use only cache, than do not make search call
            return False

        # Find partner category with this name in kb9 db
        res_ids = client[model].search(domain, context={'active_test': active_test})

        # raise error if there more than 1 partner with same ref
        if len(res_ids) > 1:
            raise TransferError(
                "More then one record found:\n"
                "\tclient:%s\n"
                "\tmodel:%s\n"
                "\tdomain=%s\n" % (client.get_url(), model, domain))

        # If ther only one partner in kb9 db, just return its ID
        if len(res_ids) == 1:
            if cached:
                mcache[key] = res_ids[0]
            return res_ids[0]

        return False

    def cache_value(self, client:Client, model:str, domain:list, res_id:int):
        """ Save (domain, res_id) pair in cache)
        """
        mcache = self.get_model_cache(client, model)
        mcache[self.get_cache_key(domain)] = res_id

    def populate_cache(self, client:Client, model:str, data,
                       domain_getter, id_getter):
        """ Allows to fill cache with some predefined values
            :param data: iterable with data to be processed
            :param domain_getter: callable wich will called for each data element to get domain
            :param id_getter: callable wich will be called for each data element to get id
        """
        msg_tmpl = (
            "%25s: cached [%%(processed)4d of %%(total)4d records] "
            "(total %%(time_total).2f sec (%%(time_per_item).2f sec/record), "
            "time remaining: %%(time_remaining).2f"
            "" % model)
        for element in in_progress(data, msg=msg_tmpl):
            self.cache_value(client, model,
                             domain_getter(element),
                             id_getter(element))


get_only_one_id = GetOnlyOneID()


def adapter(field:str):
    """ Function to mark method as adapter for specific field
    """
    def decorator(method):
        method.__adapter_for__ = field
        return method
    return decorator


class TransferModelMeta(type):
    """ Meta class to manage transfer models.
        Configure missing values and register classes
    """
    # TODO: add ability to inherit models
    #       idea - if there are already registered class for such model name,
    #       then add it as base class for newly created class
    model_classes = {}

    def __new__(mcs, name, bases, attrs):
        # add field 'field_adapters' if it is not present
        if 'field_adapters' not in attrs:
            attrs['field_adapters'] = {}

        # process attributes and find adapters,
        # and add them to 'field_adapters' dict
        for key, val in attrs.items():
            if callable(val) and getattr(val, '__adapter_for__', None):
                attrs['field_adapters'][val.__adapter_for__] = val

        # Add model_from_name and model_to_name fields if required
        if attrs.get('model', None) and attrs.get('model_from_name', None) is None:
            attrs['model_from_name'] = attrs['model']

        if attrs.get('model', None) and attrs.get('model_to_name', None) is None:
            attrs['model_to_name'] = attrs['model']

        # link_field related logic
        if attrs.get('link_field', None) and not attrs.get('link_field_source', None):
            attrs['link_field_source'] = attrs['link_field']

        if attrs.get('link_field', None) and not attrs.get('link_field_dest', None):
            attrs['link_field_dest'] = attrs['link_field']

        # create class
        cls = super(TransferModelMeta, mcs).__new__(mcs, name, bases, attrs)

        # and if it have 'model' field defined, add it to map
        if cls.model:
            mcs.model_classes[cls.model] = cls

            # also aliase this clas by 'model_from_name' and 'model_to_name'
            if cls.model_from_name and cls.model_from_name not in mcs.model_classes:
                mcs.model_classes[cls.model_from_name] = cls
            if cls.model_to_name and cls.model_to_name not in mcs.model_classes:
                mcs.model_classes[cls.model_to_name] = cls

        return cls

    def get_model_class(cls, name:str):
        """ Return TransferModel class for specific model name
        """
        return cls.model_classes[name]

    def __getitem__(cls, name:str):
        return cls.get_model_class(name)


class Transfer(object):
    """ Simple class to manage TransferModel collection,
        and do transfer itself.

        Instances of this class create TransferModel instances bound to
        pair of (source client, dest client) and plays as connector
        betwen transfer models.
    """
    def __init__(self, cl_from:Client, cl_to:Client, simplified_checks=False):
        self.cl_from = cl_from
        self.cl_to = cl_to
        self.get_only_one_id = GetOnlyOneID(cache_only=simplified_checks)
        self.cl_from_cache = empty_cache(cl_from)
        self.cl_to_cache = empty_cache(cl_to)

        self._models = {}  # here model instances will be placed

    def getModel(self, name:str):
        """ Get model instance by model name
        """
        model = self._models.get(name, None)
        if model is None:
            model_class = TransferModel[name]
            model = self._models[name] = model_class(self)

            if model.model_to_name not in self._models:
                self._models[model.model_to_name] = model

            if model.model_from_name not in self._models:
                self._models[model.model_from_name] = model

            # for some models may be enabled autopopulation of cache
            if model.auto_populate_cache:
                model.populate_cache()
        return model

    def __getitem__(self, name:str):
        return self.getModel(name)

    def __iter__(self):
        return iter((self[mname] for mname in TransferModel.model_classes))

    def auto_transfer(self):
        """ Automaticaly transfer models that have ``auto_transfer_enabled`` field set to True
        """
        auto_transfer_models = []

        # prepare models
        for model in self:
            #model = self[model_name]
            if model.auto_transfer_enabled:
                auto_transfer_models.append(model)
            #if model.auto_populate_cache:
                #model.populate_cache()   # model_cache is popuated automaticaly on model creation

        auto_transfer_models.sort(key=lambda x: x.auto_transfer_priority)

        # Transfer models
        for model in auto_transfer_models:
            model.auto_transfer()

    @property
    def stat(self):
        res = {'created': {}}
        for model in self:
            res['created'][model.model] = model.created_count
        return res


class TransferModel(object, metaclass = TransferModelMeta):
    """ Abstract class to combine transfer logic for single model

        Subclass this class and define attributes that describe data-transfer
        process

        Following attributes available:

            model - (required) model name to transfer
                    (for example 'product.product')
            model_from_name - name of model in source database
                              (used only if src and dest models are different)
            model_to_name - name of model in dest database
                            (used only if src and dest models are different)
            transfer_fields - list of fields to transfer
            renamed_fields - dictionary with field renames
                             name in src db -> name in dest db
            field_adapters - dictionary with adapter functions for fields.
                             use @adapter decorator instead
            link_field - name of field that have to have same value in
                         source and dest database
            link_field_source - name of link field in source database
                                (used only if link field is different
                                 in source and dest database)
            link_field_dest - name of link_field in destination database
                              (used only if link field is different
                               in source and dest database)
            link_field_auto_generate - automaticaly generate link field.
                                       if set to True, then custom link field
                                       will be created automaticaly on dest db
                                       useful for periodic updates
            auto_transfer_enabled - Enable autotransfer of this model.
                                    Models with this attribute set to True
                                    will be automaticaly transfered when
                                    Transfer.auto_transfer is called.
                                    Other models, may will be used only
                                    for recursive transfer of related records
            auto_transfer_domain - Domain to filter records that
                                   will be transfered by auto transfer

        Simplest example:

            from odoo_transfer_lib import Transfer, TransferModel
            from odoo_rpc_client import Client

            class TPartnerModel(TransferModel):
                model = 'res.partner'

                transfer_fields = ['name']
                link_field = 'name'
                auto_transfer_enabled = True
                auto_transfer_domain = [('parent_id', '=', False)]

            cl_from = Client(...)
            cl_to = Client(...)

            transfer = Transfer(cl_from, cl_to, simplified_checks=True)
            transfer.auto_transfer()
            print(transfer.stat['created'])
    """

    # model names
    model = None
    model_from_name = None  # if not redefined in subclass, then it will be set to value of 'model' field
    model_to_name = None    # if not redefined in subclass, then it will be set to value of 'model' field

    # TODO: decide what to do if there are no such field in source model, but
    # it is required in destination model
    transfer_fields = []    # list of fields in source model
    renamed_fields = {}     # { 'source_name': 'dest_name'}
    field_adapters = {}     # { 'field_name': lambda self, source_record: source_record[field_name]

    # link field
    # if link field is set, thent default search domain will be used as [(link_field_dest, '=', record[link_field_source])]
    link_field = None          # if set, than this field will be used to link records in both databases.
                               # this field must be unique
    # If folowing fields will not be redefined in subclass, but link_field will be redefined,
    # they will be set to be equal to link_field automaticaly (by metaclass)
    link_field_source = None   # name of link field in source model
    link_field_dest = None     # name of link field in destination model

    link_field_auto_generate = False  # if set to True, then link_field will be automaticaly generated

    # Transfer all records or only active
    auto_active_test = False

    # cache related fields
    populate_cache_domain = None
    auto_populate_cache = False

    # auto_transfer_fields
    auto_transfer_enabled = False
    auto_transfer_domain = None
    auto_transfer_priority = 10
    auto_transfer_xmlids = False   # experimental feature

    # TODO: add check if field to be written to dest is present there

    def __init__(self, transfer):
        self.transfer = transfer

        # stat fields
        self._created_ids = []

        # Check if model defined correctly
        self._check_fields()

        # Generate missing adapters for fields
        self._generate_missing_m2o_adapters()

        # Generate link field
        self._generate_link_field()

    def __new__(cls, *args, **kwargs):
        if cls.model is None:
            raise TransferError(
                "Cannot create instance when model is not specified")
        return super(TransferModel, cls).__new__(cls)

    def _generate_link_field(self, field='x_transfer_sync_id'):
        """ Generate link_field if required
        """
        if (self.link_field_auto_generate and
                not self.link_field_source and
                not self.link_field_dest):

            # test if field already present in dest database
            cl = self.cl_to
            model = self.model_to_name
            field_id = cl['ir.model.fields']([('model', '=', model),
                                              ('name', '=', field)])
            if not field_id:

                field_id =cl['ir.model.fields'].create({
                    'name': field,
                    'ttype': 'integer',
                    'state': 'manual',
                    'model_id': cl._ir_model(model=model)[0].id,
                    'model': model,
                    'field_description': field.replace('_', ' '),
                    'help': 'This field was created automaticaly during data transfer to reference data in original database',
                })
                self.dest_model._columns_info = None  # clean columns cache

            # configure link field logic to use newly created field
            self.link_field = field
            self.link_field_dest = field
            self.link_field_source = 'id'

            # adda field adapter to fill this fields with data during transfer
            self.field_adapters[field] = lambda self, record: record.id

            # configure populate cache
            if self.auto_populate_cache:
                if self.populate_cache_domain is None:
                    self.populate_cache_domain = [(field, '!=', False)]
                else:
                    self.populate_cache_domain.append((field, '!=', False))

    def _generate_missing_m2o_adapters(self):
        for field in self.get_transfer_fields():
            field_from, field_to = self._get_field_names(field)

            # There are already defined adapter for this model for this field
            # Skip it
            if field_from in self.field_adapters or field_to in self.field_adapters:
                continue

            # Get field types
            field_to_type = self.dest_model.columns_info.get(field_to, {}).get('type', False)
            field_from_type = self.source_model.columns_info.get(field_from, {}).get('type', False)

            # Add generic many2one adapter if there are no one defined yet
            if field_to_type == 'many2one' and field_from_type == 'many2one':
                self.field_adapters[field_from] = GenericM2OAdapter(field)

            # Add generic one2many adapter if there are no one defined yet
            elif field_to_type == 'one2many' and field_from_type == 'one2many':
                self.field_adapters[field_from] = GenericO2MAdapter(field)

            # Add generic many2many adapter if there ano one defined yet
            elif field_to_type == 'many2many' and field_from_type == 'many2many':
                self.field_adapters[field_from] = GenericM2MAdapter(field)

    def _check_fields(self):
        bad_fields = []
        for field in self.get_transfer_fields():
            _, field_to = self._get_field_names(field)
            if field_to not in self.dest_model.columns_info:
                bad_fields.append(field_to)
        if bad_fields:
            raise TransferError(
                "ERROR! Following fields are not present in dest databse:\n"
                "\tmodel: %s"
                "\tfields: %s" % (self.dest_model, ', '.join(bad_fields))
            )

    def _get_field_names(self, field_from:str):
        field_to = field_from   # destination field

        if field_from in self.renamed_fields:
            field_to = self.renamed_fields[field_from]

        return field_from, field_to

    @property
    def get_only_one_id(self):
        return self.transfer.get_only_one_id

    @property
    def cl_from(self):
        """ Client to source database to transfer data from
        """
        return self.transfer.cl_from

    @property
    def cl_to(self):
        """ Client to destination database to transfer data to
        """
        return self.transfer.cl_to

    @property
    def cl_from_cache(self):
        return self.transfer.cl_from_cache

    @property
    def cl_to_cache(self):
        return self.transfer.cl_to_cache

    @property
    def dest_model(self):
        """ Destination model to transfer data to

            :type: openerp_proxy.orm.object.Object
        """
        return self.cl_to[self.model_to_name]

    @property
    def source_model(self):
        """ Source model to transfer data from

            :type: openerp_proxy.orm.object.Object
        """

        return self.cl_from[self.model_from_name]

    @property
    def created_ids(self):
        return self._created_ids

    @property
    def created_count(self):
        return len(self._created_ids)

    def get_search_domain(self, record):
        """ Should return domain to search record in destination database
        """
        # if there are 'link_field' defined on model, then it is posible to use
        # simple generated domain
        if self.link_field is not None:
            if isinstance(record, Record) and record._client is self.cl_from:
                # standard case, to get domain to search for record in dest database
                return [(self.link_field_dest, '=', record[self.link_field_source])]
            elif isinstance(record, Record) and record._client is self.cl_to:
                # in this case 'get_search_domain' is called on destination
                # database's record to get domain to save record in cache
                # (domain is used as part of cache key in this case)
                return [(self.link_field_dest, '=', record[self.link_field_dest])]
            elif isinstance(record, dict):
                return [(self.link_field_dest, '=', record[self.link_field_dest])]

        raise NotImplementedError()

    def get_create_dest_context(self, source_record:Record):
        """ Simple hook to add creation context
        """
        return None

    def get_transfer_fields(self, exclude_fields=None):
        """ Returns list of fields to be transfered
        """
        fields = set(self.transfer_fields + list(self.field_adapters))
        exclude_fields = [] if exclude_fields is None else exclude_fields
        return [f for f in fields if f not in exclude_fields]

    def prepare_to_transfer(self, record:Record):
        """ Prepare record to be tranfered to destination database.
            record is bounded to client ``self.cl_from``

            Subclasses that override this method MUST call super method, to trigger this chceck
        """
        assert (record._client == self.cl_from), "Record must be bound to source client here!!!"

    def prepare_create_data_field(self, record:Record, field_from:str, field_to:str):
        if field_from in self.field_adapters:
            res = self.field_adapters[field_from](self, record)
        elif field_to in self.field_adapters:
            res = self.field_adapters[field_to](self, record)
        else:
            res = record[field_from]

        if isinstance(res, datetime.date):
            res = res.strftime('%Y-%m-%d')
        elif isinstance(res, datetime.datetime):
            res = res.strftime('%Y-%m-%d %H:%M:%S')
        return res

    def prepare_create_data(self, record:Record, exclude_fields=None):
        # prepare product_data
        data = {}
        for field in self.get_transfer_fields(exclude_fields):
            field_from, field_to = self._get_field_names(field)

            try:
                data[field_to] = self.prepare_create_data_field(record, field_from, field_to)
            except Exception:
                print("Cannot transfer field:\n"
                      "\tfield_from: %s\n"
                      "\tfield_to:   %s\n"
                      "\trecord:     %s\n"
                      "" % (field_from, field_to, record))
                raise

        return data

    def get_dest_id(self, domain):
        try:
            res_id = self.get_only_one_id(self.cl_to, self.model_to_name, domain)
        except Exception:
            print("Cannot get remote record: model: %s, domain: %s" % (self.model_to_name, domain))
            raise
        return res_id

    def create_dest_id(self, data, context=None):
        try:
            res_id = self.dest_model.create(data, context=context)
        except:
            print("Cannot create record for model '%s' with data: %s" % (self.model_to_name, data))
            raise
        return res_id

    def get_or_create_dest_id(self, record:Record):
        """ Get ID of destination record, if it had been transfered already
        """
        self.prepare_to_transfer(record)
        domain = self.get_search_domain(record)

        # get destination record if it have been transfered already
        res_id = self.get_dest_id(domain)

        # if record not found on destination database, create it
        if not res_id:
            if not self.get_transfer_fields():
                raise Exception("Cennot tranfer record for model: %s, search_domain: %s\n"
                                "No tranfer fields defined!" % self.model_from_name, self.domain)
            model_data = self.prepare_create_data(record)
            res_id = self.create_dest_id(model_data,
                                         context=self.get_create_dest_context(record))
            self.post_create_dest(record, res_id)
        return res_id

    def post_create_dest(self, record:Record, dest_id:int):
        """ Hook to modify just created records in dest database

            NOTE: overriden methods must call super method!

            :param record: record instance in source database, which was used to create
                           record in destination db
            :param dest_id: ID of created record in destination database
        """
        self._created_ids.append(dest_id)
        self.get_only_one_id.cache_value(self.cl_to,
                                         self.model_to_name,
                                         self.get_search_domain(record),
                                         dest_id)

        if self.auto_transfer_xmlids:
            xml_ids = self.cl_from.plugins.external_ids.get_for(record)
            for xml_id in xml_ids:
                if xml_id.module == '__export__' and xml_id.res_id != dest_id:
                    # do not transfer export xmlids, because thye are based on
                    # record ID, which may lead to troubles, when id in xmlid
                    # is different then object's ID
                    continue
                self.cl_to['ir.model.data'].create({
                    'model': self.model_to_name,
                    'module': xml_id.module,
                    'name': xml_id.name,
                    'res_id': dest_id,
                    'noupdate': xml_id.noupdate,
                })

    def populate_cache(self, data=None):
        if data is None:
            domain = []
            if self.populate_cache_domain is not None:
                domain = self.populate_cache_domain

            data = self.dest_model.search_records(
                domain,
                context={'active_test': self.auto_active_test},
                cache=self.cl_to_cache)

        self.get_only_one_id.populate_cache(self.cl_to,
                                            self.model_to_name,
                                            data,
                                            self.get_search_domain,
                                            lambda x: x.id)

    def get_auto_transfer_data(self):
        data = self.source_model.search_records(self.auto_transfer_domain,
                                                context={'active_test': self.auto_active_test},
                                                cache=self.cl_from_cache)

        # filter out already transfered data
        transfered_ids = data.filter(
            lambda x, self=self: self.get_dest_id(self.get_search_domain(x))).ids
        return self.source_model.search_records(self.auto_transfer_domain + [('id', 'not in', transfered_ids)],
                                                context={'active_test': self.auto_active_test},
                                                cache=self.cl_from_cache)

    def auto_transfer(self):
        if self.auto_transfer_enabled:
            # Find data
            data = self.get_auto_transfer_data()

            # Display progressbar
            # progress_msg = "Transfer '%s' [%d of %d records]" % (self.model_from_name, 0, len(data))
            # progress = IntProgress(value=0, min=0, max=len(data),
                                   # description=progress_msg)
            # display(progress)

            # Prefetch data
            data.prefetch(*self.get_transfer_fields())  # prefetch fields for faster work

            # Start transfer
            msg_tmpl = (
                "%25s: transfered [%%(processed)4d of %%(total)4d records] "
                "(total %%(time_total).2f sec (%%(time_per_item).2f sec/record), "
                "time remaining: %%(time_remaining).2f"
                "" % self.model_from_name
            )
            for record in in_progress(data, msg=msg_tmpl, close=False):
                self.get_or_create_dest_id(record)

            # progress.description = ("%25s: transfered [%4d of %4d records](new: %d)"
            # "" % (self.model_from_name, i, len(data), len(self._created_ids)))


class GenericAdapterBase(object):
    """ Generic class for field adapters
    """
    def __init__(self, field:str):
        self.field = field
        self.__adapter_for__ = field

    def __call__(self, model:TransferModel, record:Record):
        field_from, field_to = model._get_field_names(self.field)
        if record[field_from]:
            return self.process_field(model, record, field_from, field_to)
        return False

    def process_field(self, model:TransferModel, record:Record, field_from:str, field_to:str):
        raise NotImplementedError


class GenericM2OAdapter(GenericAdapterBase):
    """ Simple class of generic many2one adapter

        Usage::

            adapter = GenericM2OAdapter('partner_id')
    """
    def process_field(self, model:TransferModel, record:Record, field_from:str, field_to:str):
        rel_model_name = model.dest_model.columns_info[field_to]['relation']
        res = model.transfer[rel_model_name].get_or_create_dest_id(record[field_from])
        return res


class GenericX2MAdapter(GenericAdapterBase):
    """ Simple class of generic x2many adapter

        Usage::

            adapter = GenericX2MAdapter('order_line', 'create')
            adapter = GenericX2MAdapter('category_id', 'link')

        :param str field: name of fields this adapter is for
        :param str adapter_type: type of adapter. one of ('link', 'create')
        :param domain: standard odoo domain or callable of one argument
                       suitable for RecordList.filter method
                       It will be applied to filter related records to be transfered
    """

    def __init__(self, field:str, adapter_type:str, domain=None, **kwargs):
        assert adapter_type in ('link', 'create'), "adapter_type not in ('link', 'create')"
        super(GenericX2MAdapter, self).__init__(field, **kwargs)
        self.adapter_type = adapter_type
        self.domain = [] if domain is None else domain

    def process_field(self, model:TransferModel, record:Record, field_from:str, field_to:str):
        rel_model_name = model.dest_model.columns_info[field_to]['relation']
        rel_model_field = model.dest_model.columns_info[field_to].get('relation_field', None)
        rel_model = model.transfer[rel_model_name]
        exclude_fields = [] if rel_model_field is None else [rel_model_field]

        rel_obj_list = record[field_from]

        # if there is domain defined on adapter, then filter value by this domain
        if rel_obj_list and self.domain:
            # if domain is standard odoo domain:
            if isinstance(self.domain, (list, tuple)):
                rel_obj_list = rel_obj_list.search_records(self.domain)
            # if domain is callable, we can used filter, which in case of auto transfer should be faster
            elif callable(self.domain):
                rel_obj_list = rel_obj_list.filter(self.domain)

        # prefetch related data
        if rel_obj_list:
            rel_obj_list.prefetch(*rel_model.get_transfer_fields(exclude_fields=exclude_fields))

        res = []
        for rel_obj in rel_obj_list:
            if self.adapter_type == 'create':
                res += [
                    (0, 0,
                     rel_model.prepare_create_data(rel_obj,
                                                   exclude_fields=exclude_fields)
                    )
                ]
            elif self.adapter_type == 'link':
                res += [
                    (4, rel_model.get_or_create_dest_id(rel_obj)),
                ]
        return res


class GenericO2MAdapter(GenericX2MAdapter):
    """ Simple class of generic one2many adapter

        Usage::

            adapter = GenericO2MAdapter('order_line')
    """

    def __init__(self, field, adapter_type='create', **kwargs):
        super(GenericO2MAdapter, self).__init__(field, adapter_type, **kwargs)


class GenericM2MAdapter(GenericX2MAdapter):
    """ Simple class of generic many2many adapter

        Usage::

            adapter = GenericM2MAdapter('category_id')
    """

    def __init__(self, field, adapter_type='link', **kwargs):
        super(GenericM2MAdapter, self).__init__(field, adapter_type, **kwargs)
