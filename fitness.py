import keras
import random 
import numpy as np 
import pickle
import sklearn
from sklearn.model_selection import KFold
from dataset import load_data
from config import Config
from utils import error
from keras import backend as K 


class Database:

    def __init__(self):
        self.data = []
    
    def insert(self, individual, fitness):
        self.data.append((individual, fitness))
        
    def save(self, name):
        with open(name, "wb") as f:
            pickle.dump(self.data, f)

class Fitness:

    def __init__(self, train_name):
        
        # load train data 
        self.X, self.y = load_data(train_name)

    def evaluate_batch(self, individuals):
        scores = []
        # TODO(proste) actually no shuffling takes places
        kf = KFold(n_splits=5, random_state=42)
        for train, test in kf.split(self.X):
            X_train, X_test = self.X[train], self.X[test]
            y_train, y_test = self.y[train], self.y[test]

            input_features = keras.layers.InputLayer(Config.input_shape)
            individual_models = [
                individual.createNetwork(input_features)
                for individual in individuals
            ]
            # TODO(proste) is it intended to effectively bin model sizes?
            sizes = [(m.count_params() // 1000) for m in individual_models]

            multi_model = keras.Model(
                inputs=input_features.input,
                outputs=[
                    individual_model.output
                    for individual_model in individual_models
                ]
            )
            multi_model.compile(
                loss=Config.loss,
                optimizer=keras.optimizers.RMSprop()
            )

            multi_model.fit(
                X_train, [y_train] * len(individual_models),
                batch_size=Config.batch_size, epochs=Config.epochs, verbose=0
            )

            pred_test = multi_model.predict(X_test)
            scores.append([
                error(y_test, yy_test)
                for yy_test in pred_test
            ])

            K.clear_session()  # free resources allocated by models

        fitness = np.mean(scores, axis=0)

        return list(zip(fitness, sizes))

    def evaluate(self, individual):
        #print(" *** evaluate *** ")

        #model = individual.createNetwork()
        #return random.random(), 
         
        random.seed(42) 
        # perform KFold crossvalidation 
        kf = KFold(n_splits=5)
        scores = []
        for train, test in kf.split(self.X):   # train, test are indicies 
            X_train, X_test = self.X[train], self.X[test]
            y_train, y_test = self.y[train], self.y[test]
                
            model = individual.createNetwork()
            size = model.count_params() // 1000
            model.fit(X_train, y_train,
                      batch_size=Config.batch_size, epochs=Config.epochs, verbose=0)
            
            yy_test = model.predict(X_test)
            scores.append(error(y_test, yy_test))

            
        fitness = np.mean(scores)

        # I try this to prevent memory leaks in nsga2-keras 
        K.clear_session()



        return fitness, size
