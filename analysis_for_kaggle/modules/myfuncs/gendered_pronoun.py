import pandas as pd

import spacy
nlp = spacy.load('en_core_web_sm')


def rename_columns(train_df, test_df):
    train_df = train_df.rename(columns={'A': 'A-noun', 'B': 'B-noun'})
    test_df = test_df.rename(columns={'A': 'A-noun', 'B': 'B-noun'})
    train_df = train_df.rename(columns={'A-coref': 'A', 'B-coref': 'B'})
    return train_df, test_df


def _name_replace(s, r1, r2):
    s = str(s).replace(r1, r2)
    for r3 in r1.split(' '):
        s = str(s).replace(r3, r2)
    return s


def get_base_features(train_df, test_df):
    for df in [train_df, test_df]:
        df['section_min'] = df[
            ['Pronoun-offset', 'A-offset', 'B-offset']].min(axis=1)
        df['Pronoun-offset2'] = df['Pronoun-offset'] + df['Pronoun'].map(len)
        df['A-offset2'] = df['A-offset'] + df['A-noun'].map(len)
        df['B-offset2'] = df['B-offset'] + df['B-noun'].map(len)
        df['section_max'] = df[
            ['Pronoun-offset2', 'A-offset2', 'B-offset2']].max(axis=1)
        df['A-dist_abs'] = (df['Pronoun-offset'] - df['A-offset']).abs()
        df['B-dist_abs'] = (df['Pronoun-offset'] - df['B-offset']).abs()
        df['A-dist'] = (df['Pronoun-offset'] - df['A-offset'])
        df['B-dist'] = (df['Pronoun-offset'] - df['B-offset'])
        df['A_max'] = (df['A-offset2'] == df['section_max']).astype(int)
        df['A_min'] = (df['A-offset2'] == df['section_min']).astype(int)
        df['B_max'] = (df['B-offset2'] == df['section_max']).astype(int)
        df['B_min'] = (df['B-offset2'] == df['section_min']).astype(int)
        df['wc'] = df.apply(
            lambda r: len(
                str(r['Text'][r['section_min']: r['section_max']]).split(' ')),
            axis=1)
    return train_df, test_df


def get_nlp_features(train_df, test_df):
    def _calc_nlp_features(s, w):
        doc = nlp(str(s))
        tokens = pd.DataFrame(
            [[token.text, token.dep_] for token in doc],
            columns=['text', 'dep'])
        return len(tokens[((tokens['text'] == w) & (tokens['dep'] == 'poss'))])

    for df in [train_df, test_df]:
        df['Text'] = df.apply(
            lambda r: _name_replace(
                r['Text'], r['A-noun'], 'subjectone'), axis=1)
        df['Text'] = df.apply(
            lambda r: _name_replace(
                r['Text'], r['B-noun'], 'subjecttwo'), axis=1)
        df['A-poss'] = df['Text'].map(
            lambda x: _calc_nlp_features(x, 'subjectone'))
        df['B-poss'] = df['Text'].map(
            lambda x: _calc_nlp_features(x, 'subjecttwo'))
    return train_df, test_df


def get_train_neither(train_df, test_df):
    train_df['A'] = train_df['A'].astype(int)
    train_df['B'] = train_df['B'].astype(int)
    train_df['NEITHER'] = 1.0 - (train_df['A'] + train_df['B'])
    return train_df, test_df
