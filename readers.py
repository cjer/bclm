import pandas as pd
from conllu import parse
from collections import OrderedDict
import os
import numpy as np


BCLM_FOLDER = os.path.dirname(os.path.realpath(__file__))
DATA_FOLDER = os.path.join(BCLM_FOLDER, 'data')
YAP_OUT_FOLDER = os.path.join(DATA_FOLDER, 'yap_outputs')

TREEBANK_TOKEN_PATHS = {
                'train': os.path.join(YAP_OUT_FOLDER, 'spmrl_train_tokens.txt'),
                'dev': os.path.join(YAP_OUT_FOLDER, 'spmrl_dev_tokens.txt'),
                'test': os.path.join(YAP_OUT_FOLDER, 'spmrl_test_tokens.txt'),
                }

YAP_OUTPUT_PATHS = {
                    'seg': {
                            'train': os.path.join(YAP_OUT_FOLDER, 'spmrl_train_seg.conll'),
                            'dev': os.path.join(YAP_OUT_FOLDER, 'spmrl_dev_seg.conll'),
                            'test': os.path.join(YAP_OUT_FOLDER, 'spmrl_test_seg.conll'),
                    },
                    'map': {
                            'train': os.path.join(YAP_OUT_FOLDER, 'spmrl_train_map.conll'),
                            'dev': os.path.join(YAP_OUT_FOLDER, 'spmrl_dev_map.conll'),
                            'test': os.path.join(YAP_OUT_FOLDER, 'spmrl_test_map.conll'),
                    },
                    'dep': {
                            'train': os.path.join(YAP_OUT_FOLDER, 'spmrl_train_dep.conll'),
                            'dev': os.path.join(YAP_OUT_FOLDER, 'spmrl_dev_dep.conll'),
                            'test': os.path.join(YAP_OUT_FOLDER, 'spmrl_test_dep.conll'),
                    },
                }

LATTICES_PATHS = {
                    'train': os.path.join(YAP_OUT_FOLDER, 'spmrl_train.lattices'),
                    'dev': os.path.join(YAP_OUT_FOLDER, 'spmrl_dev.lattices'),
                    'test': os.path.join(YAP_OUT_FOLDER, 'spmrl_test.lattices'),
                }

DF_PATHS = {
            'spmrl': os.path.join(DATA_FOLDER, 'spdf_fixed.csv.gz'),
            'ud': os.path.join(DATA_FOLDER, 'uddf_fixed.csv.gz'),
            'yap_dev': os.path.join(YAP_OUT_FOLDER, 'yap_dev.csv.gz'),
            'yap_test': os.path.join(YAP_OUT_FOLDER, 'yap_test.csv.gz'),
           }


def read_dataframe(corpus, remove_duplicates=False, remove_very_similar=False, subset=None):
    df = pd.read_csv(DF_PATHS[corpus.lower()], low_memory=False)
    if subset is not None:
        df = df[df.set==subset]
    return df


def read_treebank_conllu(path, remove_duplicates=False, remove_very_similar=False,
                         expand_feats=True, expand_misc=True):
    # metadata must include sent_id (int)
    # if you want to remove duplicates or very similar, metadata must also include 
    # duplicate_sent_id and very_similar_sent_id
    with open(path, 'r', encoding='utf8') as f:
        sp_conllu = parse(f.read())
    fixed = []
    dup_to_remove = set()
    very_sim_to_remove = set()
    for tl in sp_conllu:
        if (remove_duplicates and int(tl.metadata['sent_id']) in dup_to_remove 
            or remove_very_similar and int(tl.metadata['sent_id']) in very_sim_to_remove):
            print ('skipped', tl.metadata['sent_id'])
            continue
        for tok in tl:
            t = OrderedDict(tok)
            if type(t['id']) is not tuple:
                if expand_feats:
                    if t['feats'] is not None:
                        t.update({'feats_'+f: v for f, v in t['feats'].items()})
                    del(t['feats'])
                if expand_misc:
                    if t['misc'] is not None:
                        t.update({f: v for f, v in t['misc'].items()})
                    del(t['misc'])
                t.update(tl.metadata)
                fixed.append(t)
            if remove_duplicates:
                dup_to_remove = dup_to_remove | set(eval(tl.metadata['duplicate_sent_id']))
            if remove_very_similar:
                very_sim_to_remove = dup_to_remove | set(eval(tl.metadata['very_similar_sent_id']))

    df = pd.DataFrame(fixed)
    #sent_id required
    df['sent_id'] = df.sent_id.astype(int)
    
    if global_sent_id in df.columns:
          df['global_sent_id'] = df.global_sent_id.astype(int)
    if token_id in df.columns:
          df['token_id'] = df.misc_token_id.astype(int)

    return df


def read_conll(path, add_head_stuff=False, comment='#'):
    # CoNLL file is tab delimeted with no quoting
    # quoting=3 is csv.QUOTE_NONE
    df = (pd.read_csv(path, sep='\t', header=None, quoting=3, comment=comment,
                names = ['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel', 'deps', 'misc'])
                # add sentence labels
                .assign(sent_id = lambda x: (x.id==1).cumsum())
                # replace bad root dependency tags
                .replace({'deprel': {'prd': 'ROOT'}})
               )
    
    if add_head_stuff:
        df = df.merge(df[['id', 'form', 'sent', 'upostag']].rename(index=str, columns={'form': 'head_form', 'upostag': 'head_upos'}).set_index(['sent', 'id']),
               left_on=['sent', 'head'], right_index=True, how='left')
    return df

from io import StringIO

def read_lattices(path):
    dfs = []
    for i, sent in enumerate(open(path, 'r', encoding='utf8').read().split('\n\n')):
        dfs.append(pd.read_csv(StringIO(sent), sep='\t', header=None, quoting=3, 
                               names = ['ID1', 'ID2', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'token_id'])
                  .assign(sent_id = i+1))
    
    return pd.concat(dfs).reset_index(drop=True)


flatten = lambda l: [item for sublist in l for item in sublist]


def get_feats(s):
    if s!='_' and s is not None and s is not np.nan:
        feats = OrderedDict()
        for f in s.split('|'):
            k,v = f.split('=')
            k='feats_'+k
            if k not in feats:
                feats[k] = v
            else:
                feats[k] = feats[k]+','+v
        return pd.Series(feats)
    else:
        return pd.Series()

    
def read_yap_output(treebank_set='dev', tokens_path=None, dep_path=None, map_path=None, expand_feats=False, comment=None):
    if treebank_set is not None:
        tokens_path = TREEBANK_TOKEN_PATHS[treebank_set]
        dep_path = YAP_OUTPUT_PATHS['dep'][treebank_set]
        map_path = YAP_OUTPUT_PATHS['map'][treebank_set]
        
    tokens = dict(flatten([[(str(j+1)+'_'+str(i+1), tok) for i, tok in enumerate(sent.split('\n'))]
              for j, sent in 
              enumerate(open(tokens_path, 'r').read().split('\n\n'))]))
    
    lattices = read_lattices(map_path)
    dep = read_conll(dep_path, comment=comment)
    df = (pd.concat([dep, lattices.token_id], axis=1)
          .assign(sent_tok = lambda x: x.sent_id.astype(str) + '_' + x.token_id.astype(str))
          .assign(token_str = lambda x: x.sent_tok.map(tokens))
          .drop('sent_tok', axis=1)
          )
    if expand_feats:
        df = pd.concat([df, df.feats.apply(get_feats)], axis=1).drop('feats', axis=1)
        
    return df
