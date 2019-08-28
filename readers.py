import pandas as pd
from conllu import parse
from collections import OrderedDict
import os


BCLM_FOLDER = os.path.dirname(os.path.realpath(__file__))
DATA_FOLDER = os.path.join(BCLM_FOLDER, 'data')
DF_PATHS = {'spmrl': os.path.join(DATA_FOLDER, 'spdf_fixed.csv.gz'),
            'ud': os.path.join(DATA_FOLDER, 'uddf_fixed.csv.gz'),}


def read_dataframe(corpus, remove_duplicates=False, remove_very_similar=False, subset=None):
    df = pd.read_csv(DF_PATHS[corpus.lower()], low_memory=False)
    if subset is not None:
        df = df[df.set==subset]
    return df


def read_treebank_conllu(path, remove_duplicates=False, remove_very_similar=False):
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
                if t['feats'] is not None:
                    t.update({'feats_'+f: v for f, v in t['feats'].items()})
                del(t['feats'])
                if t['misc'] is not None:
                    t.update({'misc_'+f: v for f, v in t['misc'].items()})
                del(t['misc'])
                t.update(tl.metadata)
                fixed.append(t)
            if remove_duplicates:
                dup_to_remove = dup_to_remove | set(eval(tl.metadata['duplicate_sent_id']))
            if remove_very_similar:
                very_sim_to_remove = dup_to_remove | set(eval(tl.metadata['very_similar_sent_id']))

    df = (pd.DataFrame(fixed)
          .assign(sent_id = lambda x: x.sent_id.astype(int))
          .assign(global_sent_id = lambda x: x.global_sent_id.astype(int))
          .assign(misc_token_id = lambda x: x.misc_token_id.astype(int))

         )
    return df


def read_conll(path, add_head_stuff=False):
    # CoNLL file is tab delimeted with no quoting
    # quoting=3 is csv.QUOTE_NONE
    df = (pd.read_csv(path, sep='\t', header=None, quoting=3, comment='#',
                names = ['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel', 'deps', 'misc'])
                # add sentence labels
                .assign(sent = lambda x: (x.id==1).cumsum())
                # replace bad root dependency tags
                .replace({'DEPREL': {'prd': 'ROOT'}})
               )
    
    if add_head_stuff:
        df = df.merge(df[['ID', 'FORM', 'sent', 'UPOS']].rename(index=str, columns={'FORM': 'head_form', 'UPOS': 'head_upos'}).set_index(['sent', 'ID']),
               left_on=['sent', 'HEAD'], right_index=True, how='left')
    return df


def read_lattices(path):
    df = (pd.read_csv(path, sep='\t', header=None, quoting=3, 
                names = ['ID1', 'ID2', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'misc_token_id'])
                # add sentence labels
                .assign(sent = lambda x: (x.ID1==0).cumsum())
               )
    return df

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

    
def read_yap_output(tokens_path, dep_path, map_path):
    tokens = dict(flatten([[(str(j+1)+'_'+str(i+1), tok) for i, tok in enumerate(sent.split('\n'))]
              for j, sent in 
              enumerate(open(os.path.join(yap_output_dir, tokens_path), 'r').read().split('\n\n'))]))
    lattices = read_lattices(map_path)
    dep = read_conll(dep_path)
    df = (pd.concat([dep, lattices.misc_token_id], axis=1)
          .assign(sent_tok = lambda x: x.sent.astype(str) + '_' + x.misc_token_id.astype(str))
          .assign(misc_token_str = lambda x: x.sent_tok.map(tokens))
          .drop('sent_tok', axis=1)
          )
    df = pd.concat([df, df.feats.apply(get_feats)], axis=1).drop('feats', axis=1)
    return df