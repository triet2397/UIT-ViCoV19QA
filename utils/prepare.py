from underthesea import word_tokenize

def prepare_set(df):
    ref = [list(filter(None, np.delete(i,[0,1]))) for i in df.values]
    trg = []
    src = []
    new_set = []
    
    for i in ref:
        tmp=[]
        for j in i:
            s = word_tokenize(j)
            tmp.append(s)
        trg.append(tmp)
    
    for s in df['Question'].values:
        tmp = word_tokenize(s)
        src.append(tmp)
    
    for i in range(len(trg)):
        a_data_point = [src[i],trg[i]]
        new_set.append(a_data_point)
    return new_set