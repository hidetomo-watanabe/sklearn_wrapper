import math


def translate_age(dfs, train_df):
    #######################################
    # age => age range
    # categorize after
    #######################################
    for df in dfs:
        df['AgeRange'] = [0] * len(df['Age'].values)
        for i, val in enumerate(df['Age'].values):
            df['AgeRange'].values[i] = val // 10 * 10
        del df['Age']
    return dfs


def translate_fare(dfs, train_df):
    #######################################
    # no fare => mean grouped by pclass
    #######################################
    for df in dfs:
        for i, val in enumerate(df['Fare'].values):
            if math.isnan(val):
                df['Fare'].values[i] = \
                    train_df.groupby('Pclass')['Fare'].mean()[
                        df['Pclass'].values[i]]
    return df


def translate_familystatus(dfs, train_df):
    #######################################
    # sibsp + parch == 0 => no family(0)
    # survive vs no survive in same familyname
    # => more survive(1) or less survive(2)
    # categorize after
    #######################################
    # get family name
    train_df['FamilyName'] = [''] * len(train_df['Name'].values)
    for i, val in enumerate(train_df['Name'].values):
        train_df['FamilyName'].values[i] = val.split(',')[0]
    # get family name => family status
    n2s = {}
    tmp = train_df.groupby('FamilyName')['Survived']
    for key in tmp.indices.keys():
        survived_num = tmp.sum()[key]
        no_survived_num = tmp.count()[key] - survived_num
        if survived_num > no_survived_num:
            n2s[key] = 1
        else:
            n2s[key] = 2
    # main
    for df in dfs:
        df['FamilyStatus'] = [0] * len(df['Name'].values)
        for i, val in enumerate(df['Name'].values):
            family_name = val.split(',')[0]
            # no family
            if df['SibSp'].values[i] + df['Parch'].values[i] == 0:
                continue
            # any family
            if family_name in n2s:
                df['FamilyStatus'].values[i] = n2s[family_name]
    # del family name
    del train_df['FamilyName']
    return dfs

if __name__ == '__main__':
    pass
