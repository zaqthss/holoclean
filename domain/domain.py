import math

import logging
import pandas as pd
import time
from tqdm import tqdm
import itertools
import numpy as np

from dataset import AuxTables, CellStatus


class DomainEngine:
    def __init__(self, env, dataset, cor_strength = 0.1, sampling_prob=1.0, max_sample=5):
        """
        :param env: (dict) contains global settings such as verbose
        :param dataset: (Dataset) current dataset
        :param cor_strength: (float) correlation magnitude threshold for determining domain values
            from correlated co-attributes
        :param sampling_prob: (float) probability of using a random sample if domain cannot be determined
            from correlated co-attributes
        :param max_sample: (int) maximum # of domain values from a random sample
        """
        self.env = env
        self.ds = dataset
        self.topk = env["pruning_topk"]
        self.setup_complete = False
        self.active_attributes = None
        self.domain = None
        self.total = None
        self.correlations = None
        self.cor_strength = cor_strength
        self._cor_attrs = {}
        self.sampling_prob = sampling_prob
        self.max_sample = max_sample
        self.single_stats = {}
        self.pair_stats = {}
        self.all_attrs = {}

    def setup(self):
        """
        setup initializes the in-memory and Postgres auxiliary tables (e.g.
        'cell_domain', 'pos_values').
        """
        tic = time.time()
        self.find_correlations()
        self.setup_attributes()
        domain = self.generate_domain()
        self.store_domains(domain)
        status = "DONE with domain preparation."
        toc = time.time()
        return status, toc - tic

    def find_correlations(self):
        """
        find_correlations memoizes to self.correlations a DataFrame containing
        the pairwise correlations between attributes (values are treated as
        discrete categories).
        """
        df = self.ds.get_raw_data()[self.ds.get_attributes()].copy()
        # convert dataset to categories/factors
        for attr in df.columns:
            df[attr] = df[attr].astype('category').cat.codes
        # drop columns with only one value and tid column
        df = df.loc[:, (df != 0).any(axis=0)]
        # Computer correlation across attributes
        m_corr = df.corr()
        self.correlations = m_corr

    def store_domains(self, domain):
        """
        store_domains stores the 'domain' DataFrame as the 'cell_domain'
        auxiliary table as well as generates the 'pos_values' auxiliary table,
        a long-format of the domain values, in Postgres.

        pos_values schema:
            _tid_: entity/tuple ID
            _cid_: cell ID
            _vid_: random variable ID (all cells with more than 1 domain value)
            _

        """
        if domain.empty:
            raise Exception("ERROR: Generated domain is empty.")
        else:
            self.ds.generate_aux_table(AuxTables.cell_domain, domain, store=True, index_attrs=['_vid_'])
            self.ds.aux_table[AuxTables.cell_domain].create_db_index(self.ds.engine, ['_tid_'])
            self.ds.aux_table[AuxTables.cell_domain].create_db_index(self.ds.engine, ['_cid_'])
            query = "SELECT _vid_, _cid_, _tid_, attribute, a.rv_val, a.val_id from %s , unnest(string_to_array(regexp_replace(domain,\'[{\"\"}]\',\'\',\'gi\'),\'|||\')) WITH ORDINALITY a(rv_val,val_id)" % AuxTables.cell_domain.name
            self.ds.generate_aux_table_sql(AuxTables.pos_values, query, index_attrs=['_tid_', 'attribute'])

    def setup_attributes(self):
        self.active_attributes = self.get_active_attributes()
        total, single_stats, pair_stats = self.ds.get_statistics()
        self.total = total
        self.single_stats = single_stats
        tic = time.clock()
        self.raw_pair_stats = pair_stats
        self.pair_stats = self._topk_pair_stats(pair_stats)
        toc = time.clock()
        prep_time = toc - tic
        logging.debug("DONE with pair stats preparation in %.2f secs", prep_time)
        self.setup_complete = True

    def _topk_pair_stats(self, pair_stats):
        """
        _topk_pair_stats converts 'pair_stats' which is a dictionary mapping
            { attr1 -> { attr2 -> {val1 -> {val2 -> count } } } } where
                DataFrame contains 3 columns:
                  <val1>: all possible values for attr1
                  <val2>: all values for attr2 that appeared at least once with <val1>
                  <count>: frequency (# of entities) where attr1: <val1> AND attr2: <val2>

        to a flattened 4-level dictionary { attr1 -> { attr2 -> { val1 -> [Top K list of val2] } } }
        i.e. maps to the Top K co-occurring values for attr2 for a given
        attr1-val1 pair.
        """

        out = {}
        for attr1 in tqdm(pair_stats.keys()):
            out[attr1] = {}
            for attr2 in pair_stats[attr1].keys():
                out[attr1][attr2] = {}
                for val1 in pair_stats[attr1][attr2].keys():
                    denominator = self.single_stats[attr1][val1]
                    # TODO(richardwu): while this computes tau as the topk % of
                    # count, we are actually not guaranteed any pairwise co-occurrence
                    # thresholds exceed this count.
                    # For example suppose topk = 0.1 ("top 10%") but we have
                    # 20 co-occurrence counts i.e. 20 unique val2 with
                    # uniform counts, therefore each co-occurrence count
                    # is actually 1 / 20 = 0.05 of the frequency for val1.
                    tau = float(self.topk*denominator)
                    top_cands = [val2 for (val2, count) in pair_stats[attr1][attr2][val1].items() if count > tau]
                    out[attr1][attr2][val1] = top_cands
        return out

    def get_active_attributes(self):
        """
        get_active_attributes returns the attributes to be modeled.
        These attributes correspond only to attributes that contain at least
        one potentially erroneous cell.
        """
        query = 'SELECT DISTINCT attribute as attribute FROM %s'%AuxTables.dk_cells.name
        result = self.ds.engine.execute_query(query)
        if not result:
            raise Exception("No attribute contains erroneous cells.")
        # Sort the active attributes to maintain the order of the ids of random variable.
        return sorted(itertools.chain(*result))

    def get_corr_attributes(self, attr, thres):
        """
        get_corr_attributes returns attributes from self.correlations
        that are correlated with attr with magnitude at least self.cor_strength
        (init parameter).
        """
        if (attr,thres) not in self._cor_attrs:
            self._cor_attrs[(attr,thres)] = []

            if attr in self.correlations:
                d_temp = self.correlations[attr]
                d_temp = d_temp.abs()
                self._cor_attrs[(attr,thres)] = [rec[0] for rec in d_temp[d_temp > thres].iteritems() if rec[0] != attr]
        return self._cor_attrs[(attr,thres)]

    def generate_domain(self):
        """
        Generate the domain for each cell in the active attributes as well
        as assigns variable IDs (_vid_) (increment key from 0 onwards, depends on
        iteration order of rows/entities in raw data and attributes.

        Note that _vid_ has a 1-1 correspondence with _cid_.

        See get_domain_cell for how the domain is generated from co-occurrence
        and correlated attributes.

        If no values can be found from correlated attributes, return a random
        sample of domain values.

        :return: DataFrame with columns
            _tid_: entity/tuple ID
            _cid_: cell ID (unique for every entity-attribute)
            _vid_: variable ID (1-1 correspondence with _cid_)
            attribute: attribute name
            domain: ||| seperated string of domain values
            domain_size: length of domain
            init_value: initial value for this cell
            init_value_idx: domain index of init_value
            fixed: 1 if a random sample was taken since no correlated attributes/top K values
        """

        if not self.setup_complete:
            raise Exception(
                "Call <setup_attributes> to setup active attributes. Error detection should be performed before setup.")
        # Iterate over dataset rows
        cells = []
        vid = 0
        records = self.ds.get_raw_data().to_records()
        self.all_attrs = list(records.dtype.names)
        for row in tqdm(list(records)):
            tid = row['_tid_']
            app = []
            for attr in self.active_attributes:
                init_value, init_value_idx, dom = self.get_domain_cell(attr, row)
                weak_label, fixed = self.get_weak_label(attr, row, init_value, dom, 0.99)
                weak_label_idx = dom.index(weak_label)
                if len(dom) > 1:
                    cid = self.ds.get_cell_id(tid, attr)
                    app.append({"_tid_": tid, "attribute": attr, "_cid_": cid, "_vid_":vid, "domain": "|||".join(dom),  "domain_size": len(dom),
                                "init_value": init_value, "init_index": init_value_idx, "weak_label": weak_label, "weak_label_idx": weak_label_idx, "fixed": fixed})
                    vid += 1
                else:
                    add_domain = self.get_random_domain(attr,init_value)
                    # Check if attribute has more than one unique values
                    if len(add_domain) > 0:
                        dom.extend(add_domain)
                        cid = self.ds.get_cell_id(tid, attr)
                        app.append({"_tid_": tid, "attribute": attr, "_cid_": cid, "_vid_": vid, "domain": "|||".join(dom),
                                    "domain_size": len(dom),
                                    "init_value": init_value, "init_index": init_value_idx, "weak_label": init_value, "weak_label_idx": init_value_idx, "fixed": CellStatus.single_value.value})
                        vid += 1
            cells.extend(app)

        domain_df = pd.DataFrame(data=cells).sort_values('_vid_')
        logging.info('DONE generating domain')
        return domain_df

    def get_domain_cell(self, attr, row):
        """
        get_domain_cell returns a list of all domain values for the given
        entity (row) and attribute.

        We define domain values as values in 'attr' that co-occur with values
        in attributes ('cond_attr') that are correlated with 'attr' at least in
        magnitude of self.cor_strength (init parameter).

        For example:

                cond_attr       |   attr
                H                   B                   <-- current row
                H                   C
                I                   D
                H                   E

        This would produce [B,C,E] as domain values.

        :return: (initial value of entity-attribute, domain values for entity-attribute).
        """

        domain = set()
        correlated_attributes = self.get_corr_attributes(attr, self.cor_strength)
        # Iterate through all attributes correlated at least self.cor_strength ('cond_attr')
        # and take the top K co-occurrence values for 'attr' with the current
        # row's 'cond_attr' value.
        for cond_attr in correlated_attributes:
            # Ignore correlations with index, tuple id or the same attribute.
            if cond_attr == attr or cond_attr == '_tid_':
                continue
            if not self.pair_stats[cond_attr][attr]:
                logging.warning("domain generation could not find pair_statistics between attributes: {}, {}".format(cond_attr, attr))
                continue
            cond_val = row[cond_attr]
            # Ignore correlations with null values.
            if cond_val == '_nan_':
                continue
            s = self.pair_stats[cond_attr][attr]
            try:
                candidates = s[cond_val]
                domain.update(candidates)
            except KeyError as missing_val:
                if row[attr] != '_nan_':
                    # Error since co-occurrence must be at least 1 (since
                    # the current row counts as one co-occurrence).
                    logging.error('value missing from statistics: {}'.format(missing_val))
                    raise

        # Add the initial value to the domain.
        init_value = row[attr]
        domain.add(init_value)

        # Remove _nan_ if added due to correlated attributes, only if it was not the initial value.
        if init_value != '_nan_':
            domain.discard('_nan_')

        # Convert to ordered list to preserve order.
        domain_lst = sorted(list(domain))

        # Get the index of the initial value. This should never raise a ValueError since we made sure
        # that 'init_value' was added.
        init_value_idx = domain_lst.index(init_value)

        return init_value, init_value_idx, domain_lst

    def get_random_domain(self, attr, cur_value):
        """
        get_random_domain returns a random sample of at most size
        'self.max_sample' of domain values for 'attr' that is NOT 'cur_value'.
        """
        domain_pool = set(self.single_stats[attr].keys())
        domain_pool.discard(cur_value)
        domain_pool = sorted(list(domain_pool))
        size = len(domain_pool)
        if size > 0:
            k = min(self.max_sample, size)
            additional_values = np.random.choice(domain_pool, size=k, replace=False)
        else:
            additional_values = []
        return sorted(additional_values)

    def get_weak_label(self, attr, row, init_value, dom, thres=0.99, corr_strength=0.3):
        """
        Uses a Naive Bayes model to suggest a correct value for a cell. If the confidence of the
        predicted value is above <thress> the predicted value is used as ground truth.
        :param attr: The attribute of the cell under consideration.
        :param row: The row of the cell under consideration.
        :param init_value: The initial value of cell (attr, row).
        :param dom: The pruned domain of the random variable corresponding to cell (attr, row).
        :param thres: The confidence threshold.
        :return: Returns a tuple (weak_label, fixed). The weak_label corresponds to a value from the domain and fixed
        indicates if this value should be used as ground truth or not.
        """
        nb_score = {}
        for val1 in dom:
            val1_count = self.single_stats[attr][val1]
            log_prob = math.log(float(val1_count)/float(self.total))
            correlated_attributes = self.get_corr_attributes(attr, corr_strength)
            total_log_prob = 0.0
            for at in correlated_attributes:
                if at != attr:
                    val2 = row[at]
                    val2_count = self.single_stats[at][val2]
                    val2_val1_count = 0.1
                    if val1 in self.raw_pair_stats[attr][at]:
                        if val2 in self.raw_pair_stats[attr][at][val1]:
                            val2_val1_count = max(self.raw_pair_stats[attr][at][val1][val2] - 1.0, 0.1)
                    p = float(val2_val1_count)/float(val1_count)
                    log_prob += math.log(p)
            nb_score.update({val1: log_prob})
        max_key = max(nb_score, key=nb_score.get)
        log_probability = nb_score[max_key]
        total_log_prob = 0.0
        for v in nb_score.values():
            total_log_prob += math.exp(v)
        weak_label = init_value
        fixed = CellStatus.not_set.value
        confidence = math.exp(log_probability)/total_log_prob
        if confidence > thres and max_key != init_value:
            weak_label = max_key
            fixed = CellStatus.weak_label.value
        return weak_label, fixed
