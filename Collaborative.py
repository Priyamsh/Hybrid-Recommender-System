from DataLoader import DataLoader
import numpy as np
import pandas as pd
import numba

class Collaborative:

    users=None
    unique_items=None
    rating=None
    rating1=None
    review=None

    def __init__(self,user,filename,datasize):
        x=DataLoader(filename,datasize)
        self.username=user
        self.df=x.df
        self.odf=None
        self.df1=x.df1
        self.userid=0
        self.complete_structure()

    def find_usercode(self):
        df = self.df
        userid = df.loc[df['reviewerID'] == self.username]['usercode'].unique()[0]
        self.userid=userid

    def sort_similarity(self):
        user=self.userid
        df1=self.df1
        df=self.df
        df2 = (df.groupby(by='usercode')['itemcode'].apply(list)).to_frame()
        df2 = df2.rename(columns={'itemcode': 'listofitems'})
        useritems = df2.loc[df2.index == user]['listofitems'].values[0]
        lst = df2.values
        lst = [e for sl in lst for e in sl]
        count = df.groupby(by='reviewerID').count()['asin'].max()
        vals = np.zeros((len(lst), count))
        for i in range(len(lst)):
            arr = lst[i]
            j = len(arr)
            vals[i][0:j] = arr
        present = np.isin(vals, useritems).astype(int)
        present = present.sum(axis=1)
        df2['itempresent'] = present
        df2 = df2.loc[df2['itempresent'] != 0]

        Collaborative.users = pd.DataFrame(index=df2.index.values)
        df2=df2['itempresent'].to_frame()
        Collaborative.users=pd.merge(Collaborative.users,df2,left_index=True,right_index=True,how='left')
        Collaborative.users['itempresent'] = 1 - (1 / np.exp(Collaborative.users['itempresent']))
        keys = df2.index.values
        Collaborative.users['similarity'] = np.nan
        userg = df1.get_group(user)
        item_user = df1.get_group(user)['itemcode'].values
        meanrating=df.loc[df['usercode']==user]['meanratingofuser'].unique()[0]
        for i in range(len(keys)):
            numerator = 0
            if (keys[i] != user):
                keysg = df1.get_group(keys[i])
                item_keys = df1.get_group(keys[i])['itemcode'].values
                intersection = list(set(item_keys).intersection(set(item_user)))
                for item in intersection:
                    numerator += (userg.loc[userg['itemcode'] == item]['normrating'].values[0]) * (
                    keysg.loc[keysg['itemcode'] == item]['normrating'].values[0])
                denominator = ((sum(userg['normrating'].values ** 2)) * (sum(keysg['normrating'].values ** 2))) ** (0.5)
                if (denominator != 0):
                    similarity = numerator / denominator
                    meanrating2=df.loc[df['usercode']==keys[i]]['meanratingofuser'].unique()[0]
                    denom=meanrating+meanrating2
                    if(denom!=0):
                        term=2*((meanrating2*meanrating)**2)/(denom)
                    else:
                        term=np.nan
                    epsilon=Collaborative.users.loc[keys[i]]['itempresent']
                    Collaborative.users.loc[Collaborative.users.index == keys[i]] = similarity*term*epsilon
            else:
                Collaborative.users.loc[Collaborative.users.index == keys[i]] = 1000
        Collaborative.users = Collaborative.users.sort_values(by=['similarity','itempresent'], ascending=False)
        Collaborative.users=Collaborative.users.head(30)
        Collaborative.users = Collaborative.users.reset_index()
        Collaborative.users=Collaborative.users[['index','similarity']]

    def form_pivot_table(self):
        df=self.df
        self.odf=df.copy(deep=True)
        similar_users=Collaborative.users
        index = []
        for i in range(len(df)):
            if (df.iloc[i]['usercode'] not in similar_users['index'].values):
                index.append(i)
        df = df.drop(index=index)
        self.df=df
        Collaborative.rating=pd.pivot_table(df,values=['normrating','polarity'],index='itemcode',columns='usercode')
        Collaborative.review = Collaborative.rating.isna().astype(int)
        Collaborative.rating1 = Collaborative.rating.copy(deep=True)
        Collaborative.rating['polarity'] = Collaborative.rating['polarity'].fillna(0)
        Collaborative.rating['normrating'] = Collaborative.rating['normrating'].fillna(0)

    def complete_structure(self):
        self.find_usercode()
        self.sort_similarity()
        self.form_pivot_table()