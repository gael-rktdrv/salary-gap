import pandas as pd
import numpy as np
from scipy.stats import t
from collections import Counter


class TwoIndependentSamples:

    def __init__(self, df, field, ref, cat):
        self.df = df  # The DataFrame
        self.cat = cat  # The category: categories of values in the field
        self.field = field  # The field: the series of the dataframe which will be analyzed
        self.ref = ref  # The reference: the series of the dataframe which will be used for analysis

    def get_count(self):
        """
        Get the count of the two groups
        df: the dataframe
        cat: one of the categories in field to be investigated
        field: one of the fields of the dataframe to be investigated
        ref: the reference to be used for comparison.

        e.g.: Investigating about pay gap per gender
        -> field = gender
        -> ref = salary
        -> cat = male
        """
        count = Counter(self.df[self.field])
        count_ovh = {i: j for i, j in count.items()}

        count_ovh = pd.DataFrame(list(
            count_ovh.items()), columns=[self.field, 'count']
        )

        count_mean = list(
            map(lambda row: round(row, 2), self.df.groupby(self.field)[self.ref].mean())
        )

        count_std = list(
            map(lambda row: round(row, 2), self.df.groupby(self.field)[self.ref].std())
        )

        count_ovh['mean'] = count_mean
        count_ovh['std'] = count_std
        count_ovh['var'] = count_ovh['std'].apply(lambda row: row ** 2)

        return count_ovh

    def compare_categories(self):
        """
        Compares one of the categories to the others
        cat: is the value to be compared with the remaining categories

        e.g.: Investigating about pay gap per position
        -> field = Positions
        -> ref = salary
        -> cat = CEO
        -> remaining: non-CEO
        """
        # The categories which will be compared with the chosen one
        remaining = "non-" + self.cat
        # DataFrame values
        comp_values = [self.cat, remaining]

        # DataFrame field values
        comp_count = [
            self.df[self.df[self.field] == self.cat][self.ref].count(),
            self.df[self.df[self.field] != self.cat][self.ref].count()
        ]
        comp_mean = [
            self.df[self.df[self.field] == self.cat][self.ref].mean(),
            self.df[self.df[self.field] != self.cat][self.ref].mean()
        ]
        comp_std = [
            self.df[self.df[self.field] == self.cat][self.ref].std(),
            self.df[self.df[self.field] != self.cat][self.ref].std()
        ]

        # Creating the dataframe
        comp_ovh = pd.DataFrame({
            'Keys': comp_values,
            'count': comp_count,
            'mean': comp_mean,
            'Std': comp_std
        })

        # Adding columns to the dataframe
        comp_ovh[['mean', 'Std']] = comp_ovh[['mean', 'Std']].applymap(lambda i: round(i, 2))
        comp_ovh['var'] = comp_ovh['Std'].apply(lambda i: i ** 2)  # Variance

        return comp_ovh

    @staticmethod
    def pooled_variance(size1, size2, std1, std2):
        """Return the pooled variance between two samples"""
        return ((size1 - 1) * std1 ** 2 + (size2 - 1) * std2 ** 2) / (size1 + size2 - 2)

    @staticmethod
    def std_err(pooled_var, size1, size2):
        """Returns the standard error of two samples"""
        return np.sqrt(pooled_var / size1 + pooled_var / size2)

    @staticmethod
    def degree_of_freedom(list1, list2):
        """Returns the degree of freedom of two samples"""
        return len(list1) + len(list2) - 2

    @staticmethod
    def mean_diff(mean1, mean2):
        """Returns the mean difference of two samples"""
        return mean1 - mean2

    @classmethod
    def get_t_score(cls, m1, m2, d0, pooled_var, n1, n2):
        """
        Returns the T-score from all the parameters such as means of the two samples,
        the pooled variance, the sizes of the samples
        """
        return (cls.mean_diff(m1, m2) - d0) / (np.sqrt(pooled_var * (1 / n1 + 1 / n2)))

    @staticmethod
    def get_p_value(t_score, dof):
        """Returns the p-score from T-score and degree of freedom"""
        return t.sf(abs(t_score), dof)

    def get_stats_values(self, comp_ovh, d0):
        """
        Used for comparison of two samples
        :return: the stats values pooled variance, p-value, T-value and compares with
        alpha (.1, .05, .01)
        """

        pooled_var = self.pooled_variance(
            comp_ovh['count'][0], comp_ovh['count'][1],
            comp_ovh['Std'][0], comp_ovh['Std'][1]
        )

        t_score = self.get_t_score(
            comp_ovh['mean'][0], comp_ovh['mean'][1], d0,
            pooled_var, comp_ovh['count'][0], comp_ovh['count'][1]
        )

        dof = self.degree_of_freedom(
            self.df[self.df[self.field] == self.cat][self.ref],
            self.df[self.df[self.field] != self.cat][self.ref]
        )

        # Getting the p-value from the value of T-score
        p_val = self.get_p_value(t_score, dof)

        return t_score, p_val


class SplitDataFrame:

    def __init__(self, df, field, lim_value, direction):
        """
        Return a slice of the dataframe corresponding to the parameters
        :param df: the dataframe
        :param field: the series of the dataframe, from which it will be sliced
        :param lim_value: the limit of the series value
        :param direction: either higher or lower
        """
        self.df = df
        self.field = field
        self.lim_value = lim_value
        self.direction = direction

    def return_dataframe(self):
        """Returns the corresponding dataframe"""
        if self.direction == 'higher':
            return self.df[self.df[self.field] > self.lim_value]
        elif self.direction == 'lower':
            return self.df[self.df[self.field] < self.lim_value]
