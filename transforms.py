import pandas as pd


TOK_FIELDS = ['sent_id', 'token_id', 'token_str']


def get_token_biose(df, biose_field, token_fields = TOK_FIELDS):
    
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
          .groupby(token_fields)
          .apply(_single_token_conversion)
          .reset_index().rename(columns={0: biose_field})
          .set_index(token_fields)
         )
    return df


def get_token_df(df, fields=None, biose=None, token_fields = TOK_FIELDS, sep='^', fill_value='', add_set=True):
    tok_dfs = []
    
    if biose is not None:
        for col in biose:
            tok_dfs.append(get_token_biose(df, col))
        
    if fields is not None:
        concat_fields = lambda x: pd.Series({f: sep.join(x[f].fillna(fill_value).tolist()) for f in fields})
        tok_fields = (df
                .groupby(token_fields)
                .apply(concat_fields))
        tok_dfs.append(tok_fields)
        
    tok_df = pd.concat(tok_dfs, axis=1)

    if add_set and 'set' in df.columns:
            tok_df = tok_df.assign(set = lambda x: (x.index
                                                     .get_level_values('sent_id')
                                                     .map(df[['sent_id', 'set']]
                                                     .drop_duplicates()
                                                     .set_index('sent_id')['set'])))
            
    tok_df = tok_df.sort_index().reset_index()
    
    return tok_df


def get_sentences_list(df, fields, sent_id='sent_id'):
    return df.groupby(sent_id)[fields].apply(lambda x: (x.values.tolist()))


def get_feature_lists(df, fields, sent_id='sent_id'):
    feats = []
    for field in fields:
        feats.append(df.groupby(sent_id)[field].apply(lambda x: (x.values.tolist())))
    return feats
