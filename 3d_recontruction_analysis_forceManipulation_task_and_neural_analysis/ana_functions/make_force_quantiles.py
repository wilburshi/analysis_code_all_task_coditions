def make_force_quantiles(yyy):
    
    import numpy as np
    import scipy.stats as st
    
    if np.shape(np.unique(yyy))[0] == 1:
        yyy_quant = np.ones(np.shape(yyy))*2
    # two kinds of force
    elif np.shape(np.unique(yyy))[0] == 2:
        ranks = st.rankdata(yyy, method='average')  # Average ranks for ties
        yyy_quant = (np.ceil(ranks / len(yyy) * 2)-1)*2+1 # separate into three quantiles
        # yyy_quant = (np.ceil(ranks / len(yyy) * 2)) # separate into two quantiles         
    # more than two kinds of force,
    else:
        ranks = st.rankdata(yyy, method='average')  # Average ranks for ties
        yyy_quant = np.ceil(ranks / len(yyy) * 3) # separate into three quantiles
        # yyy_quant = (np.ceil(ranks / len(yyy) * 2)) # separate into two quantiles

    return yyy_quant
