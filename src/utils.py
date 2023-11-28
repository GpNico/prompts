

import pandas as pd


def get_template_from_lama(df: pd.core.frame.DataFrame) -> dict:
        """
            Retrieve templats from a LAMA type dataframe.
            
            Rk: remove [X] and [Y]. Could be interesting to replace it by instantaiations.
        """
        templates = {}
        for k in range(len(df)):
            elem = df.iloc[k]
            rela = elem['predicate_id']
            if rela in templates.keys():
                continue
            templates[rela] = elem['template'].replace('[X] ', '').replace(' [Y]', '').replace(' .', '')
        return templates