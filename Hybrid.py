from Hybridmodel import Hybridmodel
import warnings
warnings.filterwarnings('ignore')

class Hybrid:

    def __init__(self,user,top_n):
        x = Hybridmodel(user,top_n,'elecfinal3.csv',500000)
        print('Hello ',user)
        print(x.hybrid)


y=Hybrid('AAP7PPBU72QFM',20)

#AZZX23UGJGKTT
#AAP7PPBU72QFM