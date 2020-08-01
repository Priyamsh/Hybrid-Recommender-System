import pandas as pd
import numpy as np

class DataLoader:

    def __init__(self,filename,datasize):
        df=pd.read_csv(filename)
        df=df.head(datasize)
        df=df.rename(columns={'overall_x':'rating'})
        df['meanvals']=np.nan
        df.loc[df['fake']==0,'meanvals']=df.loc[df['fake']==0]['rating'].mean()
        df.loc[df['fake']==2, 'meanvals'] = df.loc[df['fake'] == 2]['rating'].mean()
        df['baseline']=df['meanratingofuser']+df['avgratingofprod']-df['meanvals']
        df['normrating'] = df['rating'] - df['baseline']
        df = df[['reviewerID', 'asin','rating', 'polarity', 'subjectivity', 'meanratingofuser', 'normrating', 'itemcode',
                 'usercode','baseline','avgratingofprod']]
        df = df.groupby(['reviewerID', 'asin']).mean()
        df = df.reset_index()
        df1 = df.groupby(by='usercode')
        self.df=df
        self.df1=df1

