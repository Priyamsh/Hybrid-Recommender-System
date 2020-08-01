from Hybrid import Hybrid
from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class recommender(Resource):

    def get(self,user,top_n):
        return {'data': Hybrid.getrecc(user,top_n)}

api.add_resource(recommender, '/recommend/<user>/<top_n>')

if __name__ == '__main__':
     app.run()