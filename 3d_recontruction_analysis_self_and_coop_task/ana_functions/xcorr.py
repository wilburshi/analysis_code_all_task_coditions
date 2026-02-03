# cross correlation between two time series signals
# aim to find the best time lag shift

def xcorr(x,y,bestlag_limit):
    
    import scipy.signal as signal
    import scipy.stats as st
    import numpy as np

    """
    Perform Cross-Correlation on x and y
    x    : 1st signal
    y    : 2nd signal
    bestlag_limit: a limit for finding the bestlag_limit, in the unit of second

    returns
    lags : lags of correlation
    corr : coefficients of correlation
    """
    corr_signal = signal.correlate(x, y, mode="full")

    lags = signal.correlation_lags(len(x), len(y), mode="full")
   
    corr_signal = abs(corr_signal)
 
    ind_bestlag = np.where(corr_signal==np.max(corr_signal[(lags<bestlag_limit)&(lags>-bestlag_limit)]))[0][0]
    bestlag = lags[ind_bestlag]
    
    return lags, corr_signal, bestlag




### alternative methods
## customized correlagram 

def correlagram(x,y,bestlag_limit):
    
    import scipy.stats as st
    import numpy as np
    
    lags = np.arange(-bestlag_limit,bestlag_limit,1)
    
    nlags = np.shape(lags)[0]
    
    corrs = np.nan*np.ones(np.shape(lags))
    
    ndatasize = np.shape(x)[0]
    
    x_rightpad = np.concatenate([np.array(x),np.zeros((1,bestlag_limit))[0]])
    x_leftpad  = np.concatenate([np.zeros((1,bestlag_limit))[0],np.array(x)])
    y_rightpad = np.concatenate([np.array(y),np.zeros((1,bestlag_limit))[0]])
    y_leftpad  = np.concatenate([np.zeros((1,bestlag_limit))[0],np.array(y)])
    
    x_bothpad = np.concatenate([np.zeros((1,bestlag_limit))[0],np.array(x),np.zeros((1,bestlag_limit))[0]])
    y_bothpad = np.concatenate([np.zeros((1,bestlag_limit))[0],np.array(y),np.zeros((1,bestlag_limit))[0]])
    
    
    for ilag in np.arange(0,nlags,1):
        if lags[ilag] < 0:
            try:
                corrs[ilag],_ = st.spearmanr(x[0:lags[ilag]],y[-lags[ilag]:])
                corrs[ilag] = corrs[ilag]*np.shape(x[0:lags[ilag]])[0]
                # corrs[ilag],_ = st.spearmanr(x_rightpad[-lags[ilag]:],y_leftpad[-lags[ilag]:])
            except:
                corrs[ilag] = np.nan
        elif lags[ilag] == 0:
            try:
                corrs[ilag],_ = st.spearmanr(x,y)
                corrs[ilag] = corrs[ilag]*np.shape(x)[0]
                # corrs[ilag],_ = st.spearmanr(x_bothpad,y_bothpad)
            except:
                corrs[ilag] = np.nan
            
        else:
            try:
                corrs[ilag],_ = st.spearmanr(y[0:-lags[ilag]],x[lags[ilag]:])
                corrs[ilag] = corrs[ilag]*np.shape(x[lags[ilag]:])[0]
                # corrs[ilag],_ = st.spearmanr(x_leftpad[lags[ilag]:],y_rightpad[lags[ilag]:])
            except:
                corrs[ilag] = np.nan
        
    ind_bestlag = np.where(abs(corrs)==np.nanmax(abs(corrs)))[0][0]
    bestlag = lags[ind_bestlag]
    
    return lags, corrs, bestlag
    
