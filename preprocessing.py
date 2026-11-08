import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    def fit(self, df):
        self.columns = list(df.columns)
        self.num_cols = [c for c in df.columns if is_numeric_dtype(df[c]) and not is_bool_dtype(df[c])]
        self.cat_cols = [c for c in df.columns if c not in self.num_cols]

        # numeric: median + standardisation
        self.medians = df[self.num_cols].median().fillna(0.0)
        if self.num_cols:
            filled = df[self.num_cols].astype(float).fillna(self.medians)
            self.scaler = StandardScaler().fit(filled.to_numpy())

        # categorical: most frequent value + list of categories
        self.categories, self.modes = {}, {}
        for c in self.cat_cols:
            values = df[c].dropna().astype(str)
            self.categories[c] = sorted(values.unique())
            self.modes[c] = values.mode().iloc[0] if len(values) else ""

        self.feature_names_out = list(self.num_cols) + [
            f"{c} = {k}" for c in self.cat_cols for k in self.categories[c]
        ]
        self.n_features_out = len(self.feature_names_out)
        return self

    def transform(self, df):
        parts = []
        if self.num_cols:
            num = df[self.num_cols].astype(float).fillna(self.medians)
            parts.append(self.scaler.transform(num.to_numpy()))
        for c in self.cat_cols:
            col = df[c].astype("object")
            col = col.where(df[c].notna(), self.modes[c]).astype(str)
            for k in self.categories[c]:
                parts.append((col == k).to_numpy(float).reshape(-1, 1))
        return np.hstack(parts)