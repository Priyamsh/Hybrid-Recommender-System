from LatentRecommendation import LatentRecommendation
from CollaborativeRecommendation import CollaborativeRecommendation
import pandas as pd
import datetime

class Hybridmodel:

    def __init__(self,user,top_n,filename,datasize):
        print(datetime.datetime.now())
        Latent1=LatentRecommendation(user,top_n,filename,datasize)
        print("Latent done")
        print(datetime.datetime.now())
        Colab1=CollaborativeRecommendation(user,top_n,filename,datasize)
        print("Colab done")
        print(datetime.datetime.now())
        self.latrating=LatentRecommendation.itemrating
        self.colrating=CollaborativeRecommendation.itemslist
        self.top_n=top_n
        self.hybrid=None
        self.get_hybrid_recc()
        self.rmse=Latent1.test_error
        self.rmse_train=Latent1.train_error

    def gettotal(self,x):
        if (x.rating_x != x.rating_x):
            return x.rating_y
        elif (x.rating_y != x.rating_y):
            return x.rating_x
        else:
            return (x.rating_x + x.rating_y) / 2

    def get_hybrid_recc(self):
        itemrating=self.latrating
        userrecc=self.colrating
        hybrid = pd.merge(itemrating, userrecc, on='item', how='outer')
        hybrid['averagescore'] = hybrid.apply(lambda x: self.gettotal(x), axis=1)
        hybrid = hybrid.sort_values(by='averagescore', ascending=False)
        hybrid=hybrid.head(self.top_n)
        hybrid=hybrid[['item','averagescore']]
        self.hybrid=hybrid

