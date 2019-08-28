import pandas as pd


TOK_FIELDS = ['sent_id', 'token_id', 'token_str']


def get_token_biose(df, biose_field):
    
    def _single_token_conversion(tok):
        all_bio = tok.biose_only.tolist()
        all_typ = set(tok.ner_type.dropna().tolist())
        if len(all_typ)>1:
            return 'O'
        if 'S' in all_bio:
            new_bio = 'S'
        elif 'B' in all_bio and 'E' in all_bio:
            new_bio = 'S'
        elif 'B' in all_bio:
            new_bio = 'B'
        elif 'E' in all_bio:
            new_bio = 'E'
        elif 'I' in all_bio:
            new_bio = 'I'
        else:
            return 'O'
        return new_bio+'-'+all_typ.pop()
    
    df[['biose_only', 'ner_type']] = df[biose_field].str.split('-', expand=True)
    df = (df
          .groupby(TOK_FIELDS)
          .apply(_single_token_conversion)
          .reset_index().rename(columns={0: biose_field})
          .set_index(TOK_FIELDS)
         )
    return df


def get_token_df(df, fields=None, biose=['biose_layer0'], sep='^', fill_value=''):
    tok_dfs = []
    
    for col in biose:
        tok_dfs.append(get_token_biose(df, col))
        
    if fields is not None:
        concat_fields = lambda x: pd.Series({f: sep.join(x[f].fillna(fill_value).tolist()) for f in fields})
        tok_fields = (df
                .groupby(TOK_FIELDS)
                .apply(concat_fields))
        tok_dfs.append(tok_fields)
        
    tok_df = (pd.concat(tok_dfs, axis=1)
                .sort_index()
                .assign(set = lambda x: (x.index
                                         .get_level_values('sent_id')
                                         .map(df[['sent_id', 'set']]
                                              .drop_duplicates()
                                              .set_index('sent_id')['set'])))
                .reset_index()
                                 )
    return tok_df