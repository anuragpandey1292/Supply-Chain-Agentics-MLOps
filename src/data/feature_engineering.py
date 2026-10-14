import pandas as pd
import numpy as np

class AdaptiveFeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.metadata = self._infer_metadata()

    def _infer_metadata(self) -> dict:
        """Autonomously maps column roles based on naming patterns."""
        cols = self.df.columns.str.lower()
        
        date_col = next((self.df.columns[i] for i, c in enumerate(cols) if 'date' in c or 'time' in c), None)
        target_col = next((self.df.columns[i] for i, c in enumerate(cols) if 'sale' in c or 'demand' in c or 'qty' in c), None)
        
        # Identify grouping granularities (SKUs, Stores, Families)
        hierarchy_cols = [self.df.columns[i] for i, c in enumerate(cols) if any(k in c for k in ['store', 'sku', 'item', 'family', 'warehouse', 'region'])]
        
        # Identify exogenous factors (promotions, holidays, weather)
        exo_cols = [self.df.columns[i] for i, c in enumerate(cols) if any(k in c for k in ['promo', 'holiday', 'price', 'oil'])]

        return {
            "date_col": date_col,
            "target_col": target_col,
            "hierarchy_cols": hierarchy_cols,
            "exo_cols": exo_cols
        }

    def transform(self) -> pd.DataFrame:
        meta = self.metadata
        df = self.df
        print(f"🔍 Inferred Metadata -> Date: {meta['date_col']} | Target: {meta['target_col']} | Granularities: {meta['hierarchy_cols']}")

        # 1. Temporal Features (If date exists)
        if meta["date_col"]:
            df[meta["date_col"]] = pd.to_datetime(df[meta["date_col"]], errors='coerce')
            df = df.sort_values(by=meta["date_col"])
            df['year'] = df[meta["date_col"]].dt.year
            df['month'] = df[meta["date_col"]].dt.month
            df['dayofweek'] = df[meta["date_col"]].dt.dayofweek
            df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

        # 2. Hierarchical Lag & Rolling Features (Adapts to whatever SKUs/Families exist)
        if meta["target_col"]:
            target = meta["target_col"]
            if meta["hierarchy_cols"]:
                # Grouped lag to prevent cross-SKU/Store data leakage
                df['lag_7'] = df.groupby(meta["hierarchy_cols"])[target].shift(7)
                df['rolling_mean_7'] = df.groupby(meta["hierarchy_cols"])[target].transform(lambda x: x.rolling(7).mean())
            else:
                # Global lag if no granularities exist
                df['lag_7'] = df[target].shift(7)
                df['rolling_mean_7'] = df[target].rolling(7).mean()

        # 3. Handle Exogenous Variables dynamically
        for exo in meta["exo_cols"]:
            df[exo] = df[exo].fillna(0)
            # Create interaction features if promotions/holidays exist
            if 'promo' in exo.lower() and meta["target_col"]:
                df['promo_lag_1'] = df.groupby(meta["hierarchy_cols"])[exo].shift(1) if meta["hierarchy_cols"] else df[exo].shift(1)

        return df.fillna(0)