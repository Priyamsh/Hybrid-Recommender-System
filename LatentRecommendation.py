from LatentBased import LatentBased
import numpy as np
import pandas as pd
import joblib

class LatentRecommendation(LatentBased):

    model=None
    itemrating=None

    def __init__(self,user,top_n,filename,datasize):
        super().__init__(filename,datasize)
        #LatentRecommendation.model=joblib.load('latentmodel.sav')
        #print(LatentRecommendation.model.rank)
        self.username=user
        self.userid=0
        self.userfeat=None
        self.itemfeat=None
        self.top_n=top_n
        self.complete_recommendation()

    def find_usercode(self):
        df1 = self.df1
        userid = df1.loc[df1['reviewerID'] == self.username]['usercode'].unique()[0]
        self.userid=userid

    def get_factor_matrices(self):
        model=LatentBased.model
        self.userfeat=model.userFactors.toPandas()
        self.itemfeat=model.itemFactors.toPandas()

    def calculate_ratings(self):
        userfeat=self.userfeat
        itemfeat=self.itemfeat
        best_model=LatentBased.model
        userid=self.userid
        top_n=self.top_n
        arr = np.array(userfeat[userfeat['id'] == userid]['features'].values[0])
        arr1 = np.zeros((len(itemfeat['features'].values), best_model.rank))
        for i in range(len(itemfeat['features'].values)):
            arr1[i, :] = itemfeat['features'].values[i]
        rating = arr1 @ arr.reshape(best_model.rank, 1)
        LatentRecommendation.itemrating = pd.DataFrame(rating, columns=['rating'])
        LatentRecommendation.itemrating['id'] = itemfeat['id']
        LatentRecommendation.itemrating = LatentRecommendation.itemrating.sort_values(by='rating', ascending=False)
        LatentRecommendation.itemrating = LatentRecommendation.itemrating.head(10 * top_n)

    def checkifbought (self,x,userid,fd):
        return len(fd.loc[(fd['usercode']==userid) & (fd['itemcode']==x)])

    def add_bought(self):
        LatentRecommendation.itemrating['bought'] = LatentRecommendation.itemrating['id'].apply(lambda x: self.checkifbought(x, self.userid, self.df1))
        LatentRecommendation.itemrating = LatentRecommendation.itemrating.loc[LatentRecommendation.itemrating['bought'] == 0]

    def add_asin_column(self):
        fd=self.df1
        LatentRecommendation.itemrating['item'] = LatentRecommendation.itemrating['id'].apply(lambda x: fd.loc[fd['itemcode'] == x]['asin'].unique()[0])
        LatentRecommendation.itemrating = LatentRecommendation.itemrating[['item', 'rating']]

    def limit_ratings(self):
        df=self.df1
        itemrating=LatentRecommendation.itemrating
        itemrating['rating'] = 5 * (itemrating['rating'] - itemrating['rating'].min()) / (
                    itemrating['rating'].max() - itemrating['rating'].min())
        LatentRecommendation.itemrating=itemrating

    def complete_recommendation(self):
        self.find_usercode()
        self.get_factor_matrices()
        self.calculate_ratings()
        self.add_bought()
        self.add_asin_column()
        self.limit_ratings()