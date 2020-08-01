from DataLoader import DataLoader
import findspark
findspark.init()
from pyspark import SparkContext
sc=SparkContext()
from pyspark.sql import SparkSession
spark=SparkSession(sc)
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
import numpy as np
import joblib

class LatentBased:
    als=None
    param_grid=None
    evaluator=None
    tvs=None
    model=None

    def __init__(self,filename,datasize):
        x=DataLoader(filename,datasize)
        self.df=x.df
        self.df1=x.df.copy(deep=True)
        self.train_data=None
        self.test_data=None
        self.train_error=0
        self.test_error=0
        self.complete_model()
        #filenames='latentmodel.sav'
        #joblib.dump(LatentBased.model,filenames)

    def pre_process_data(self):
        self.df = self.df[['itemcode', 'usercode', 'rating']].rename(columns={'itemcode': 'item', 'usercode': 'user'})

    def pyspark_file(self):
        self.df.to_csv('convertedforpyspark.csv', header=False, index=False)
        df1 = sc.textFile('convertedforpyspark.csv').map(lambda x: x.split(',')).toDF(['item', 'user', 'rating'])
        df1 = df1.withColumn('user', df1['user'].cast('int'))
        df1 = df1.withColumn('item', df1['item'].cast('int'))
        df1 = df1.withColumn('rating', df1['rating'].cast('float'))
        self.df=df1

    def split_data(self):
        (training, test) = self.df.randomSplit([0.8, 0.2])
        self.train_data=training
        self.test_data=test

    def model_building(self):
        LatentBased.als = ALS(userCol='user', itemCol='item', ratingCol='rating', coldStartStrategy='drop', nonnegative=True,
                  seed=np.random.randint(1))
        LatentBased.param_grid = ParamGridBuilder().addGrid(LatentBased.als.rank, [45]).addGrid(LatentBased.als.maxIter, [15]).addGrid(LatentBased.als.regParam,
                                                                                                   [0.18]).build()
        LatentBased.evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
        LatentBased.tvs = TrainValidationSplit(estimator=LatentBased.als, estimatorParamMaps=LatentBased.param_grid, evaluator=LatentBased.evaluator)

    def training_model(self):
        model= LatentBased.tvs.fit(self.train_data)
        best_model = model.bestModel
        LatentBased.model=best_model

    def testing_model(self):
        predictions = LatentBased.model.transform(self.test_data)
        self.test_error = LatentBased.evaluator.evaluate(predictions)
        trainingset = LatentBased.model.transform(self.train_data)
        self.train_error = LatentBased.evaluator.evaluate(trainingset)

    def complete_model(self):
        self.pre_process_data()
        self.pyspark_file()
        self.split_data()
        self.model_building()
        self.training_model()
        self.testing_model()
