from Collaborative import Collaborative
import numpy as np
import pandas as pd

class CollaborativeRecommendation(Collaborative):

    unique_items=None
    itemslist=None

    def __init__(self,user,top_n,filename,datasize):
        super().__init__(user,filename,datasize)
        self.top_n=top_n
        self.complete_reccomendation(3)

    def unique_items(self):
        df=self.odf
        CollaborativeRecommendation.unique_items = df['itemcode'].unique()

    def findkusers(self,itemcode,k):
        similar_users=Collaborative.users
        review=Collaborative.review['normrating']
        rating=Collaborative.rating
        listofusers = []
        similarityofusers = []
        normratingofusers = []
        j = 0
        for i in range(len(similar_users)):
            usercode = similar_users['index'].values[i]
            if (itemcode in review[usercode].index and review[usercode][itemcode] == 0):
                listofusers.append(usercode)
                similarityofusers.append(similar_users.loc[similar_users['index'] == usercode]['similarity'].values[0])
                normratingofusers.append(rating['normrating'][usercode][itemcode])
                j += 1
            else:
                continue
            if (j >= k):
                break
        return listofusers,similarityofusers,normratingofusers

    def rateallitems(self,k):
        usercode=self.userid
        review=Collaborative.review['normrating']
        uniitems=CollaborativeRecommendation.unique_items
        itemslist = pd.DataFrame(index=uniitems)
        itemslist['rating'] = np.nan
        m = len(itemslist)
        normratings = np.zeros((m, 3))
        similarities = np.zeros((m, 3))
        i = 0
        for i in range(len(itemslist)):
            itemtoberatedcode = itemslist.index.values[i]
            if (itemtoberatedcode not in review[usercode].index or review[usercode][itemtoberatedcode] == 1):
                listofusers, similarityofusers, normratingofusers = self.findkusers(itemtoberatedcode,k)
                n = len(listofusers)
                normratings[i, 0:n] = normratingofusers
                similarities[i, 0:n] = similarityofusers
                #if (len(listofusers) != 0):
                    #numerator = np.sum(np.multiply(similarityofusers, normratingofusers))
                    #denominator = np.sum(similarityofusers)
                    #if (denominator != 0):
                        #ratingforitem = numerator / denominator
                        #itemslist.iloc[i] = ratingforitem
                    #else:
                        #continue
                #else:
                    #continue
            else:
                continue
        numerator = np.sum(normratings * similarities, axis=1)
        denominator = np.sum(similarities, axis=1)
        ratings = numerator / denominator
        ratings[ratings == np.inf] = np.nan
        itemslist['rating'] = ratings
        itemslist = itemslist.dropna()
        itemslist = itemslist.sort_values(by='rating', ascending=False)
        CollaborativeRecommendation.itemslist=itemslist

    def recommend_products(self):
        df=self.odf
        CollaborativeRecommendation.itemslist = CollaborativeRecommendation.itemslist.reset_index()
        CollaborativeRecommendation.itemslist['asin'] = CollaborativeRecommendation.itemslist['index'].apply(lambda x: df.loc[df['itemcode'] == x]['asin'].unique()[0])

    def limit_ratings(self):
        userrecc=self.itemslist
        userrecc = userrecc.dropna()
        fd=self.odf
        userrecc['avgrating']=userrecc['index'].apply(lambda x: fd.loc[fd['itemcode']==x]['avgratingofprod'].unique()[0])
        userrecc['rating'] += userrecc['avgrating']
        userrecc = userrecc.sort_values(by='rating', ascending=False)
        userrecc=userrecc.head(10*self.top_n)
        userrecc = userrecc[['asin', 'rating']].rename(columns={'asin': 'item'})
        userrecc['rating'] = 5 * (userrecc['rating'] - userrecc['rating'].min()) / (
                    userrecc['rating'].max() - userrecc['rating'].min())
        CollaborativeRecommendation.itemslist=userrecc

    def complete_reccomendation(self,k):
        self.unique_items()
        self.rateallitems(k)
        self.recommend_products()
        self.limit_ratings()