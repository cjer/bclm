import sys
import pandas as pd
from .readers import read_conll, read_dataframe, read_yap_output, read_treebank_conllu
from collections import Counter


FORM_POS_FEATS_WITH_TOKEN_ID = ['sent_id', 'token_id', 'form', 'upostag', 'feats']
FORM_POS_WITH_TOKEN_ID = ['sent_id', 'token_id', 'form', 'upostag']
DEFAULT_COLS = FORM_POS_WITH_TOKEN_ID


def evaluate_multi_sets(gold_set, pred_set, verbose=True, examples=5):
    correct = pred_set & gold_set

    prec =   100*sum(correct.values()) / sum(pred_set.values())
    recall = 100*sum(correct.values()) / sum(gold_set.values())
    f1 = 2*prec*recall/(prec+recall)
    
    if verbose:
        print(sum(gold_set.values()), 'gold tokens/morphems,', sum(pred_set.values()), 'predicted,', sum(correct.values()), 'correct.')
        print('Precision:', round(prec, 2))
        print('Recall:   ', round(recall, 2))
        print('F1:       ', round(f1, 2))
        print('FP ex.:', [e for e in list(pred_set-gold_set)[:examples]])
        print('FN ex.:', [e for e in list(gold_set-pred_set)[:examples]])
        
    return prec, recall, f1


def create_multi_set_from_df(df, cols):
    return Counter(df.groupby(cols).size().to_dict())


def evaluate_dfs(gold_df, pred_df, cols=DEFAULT_COLS,
                 sentence_subset=None, replace_upos_tag_underscore=True):
    
    if sentence_subset is not None:
        gold_df = gold_df[gold_df.sent_id.isin(sentence_subset)]
        pred_df = pred_df[pred_df.sent_id.isin(sentence_subset)]
    
    if replace_upos_tag_underscore:
        gold_df['upostag'] = gold_df.upostag.str.replace('_','-')
        pred_df['upostag'] = pred_df.upostag.str.replace('_','-')

    gold_multi_set = create_multi_set_from_df(gold_df, cols)
    pred_multi_set = create_multi_set_from_df(pred_df, cols)
    
    return evaluate_multi_sets(gold_multi_set, pred_multi_set)


def evaluate_conllu_files(gold_conllu_path, pred_conllu_path,
                           cols = DEFAULT_COLS,
                           sentence_subset=None,
                           replace_upos_tag_underscore=True,):
    
    gold_df = read_treebank_conllu(gold_conllu_path, expand_misc=False, expand_feats=False)
    pred_df = read_treebank_conllu(pred_conllu_path, expand_misc=False, expand_feats=False)
    
    return evaluate_dfs(gold_df, pred_df, cols, sentence_subset, replace_upos_tag_underscore)
    
    
def evaluate_treebank_files(treebank_gold_set='dev', yap_pred_set='dev', 
                   cols = DEFAULT_COLS,
                   alternative_pred_fields=None,
                   truncate=None,
                   sentence_subset=None,
                   replace_upos_tag_underscore=True,
                  ):
    
    if treebank_gold_set is not None:
        gold_df = (read_dataframe('spmrl', subset=treebank_gold_set)
                   .assign(sent_id = lambda x: x.sent_id - x.sent_id.min() + 1))
    if yap_pred_set is not None:
        if yap_pred_set=='gold':
            pred_df = gold_df.copy()
        else:    
            pred_df = read_yap_output(yap_pred_set)
        
        if alternative_pred_fields is not None:
            if truncate is not None:
                old_len=pred_df.shape[0]
                pred_df = pred_df[pred_df.id<=truncate]
                print(f'truncated from {old_len} to {pred_df.shape[0]}')
            for f, alt_values in alternative_pred_fields.items():
                pred_df[f] = alt_values


    return evaluate_dfs(gold_df, pred_df, cols, sentence_subset, replace_upos_tag_underscore)
        


if __name__ == '__main__':
    gold_conllu_path, pred_conllu_path = sys.argv[1], sys.argv[2]
    
    gold_df = read_treebank_conllu(gold_conllu_path, expand_misc=False, expand_feats=False)
    pred_df = read_treebank_conllu(pred_conllu_path, expand_misc=False, expand_feats=False)
    
    evaluate_dfs(gold_df, pred_df, cols, sentence_subset, replace_upos_tag_underscore)
    