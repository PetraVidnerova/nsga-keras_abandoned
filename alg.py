import random
import pickle
import datetime

from deap import algorithms


def myNSGASimple(population, start_gen, toolbox, cxpb, mutpb, ngen,
                 stats, halloffame, logbook, verbose, id=None):

    total_time = datetime.timedelta(seconds=0)

    for gen in range(start_gen, ngen):
        start_time = datetime.datetime.now()

        offspring = algorithms.varAnd(population, toolbox, cxpb=cxpb,
                                      mutpb=mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        population = toolbox.select(population+offspring, k=len(population))

        # halloffame.update(offspring) # nsga2 halloffame works how?
        # update statics
        record = stats.compile(population)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # save checkpoint
        if gen % 1 == 0:
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(population=population, generation=gen,
                      halloffame=halloffame,
                      logbook=logbook, rndstate=random.getstate())
            if id is None:
                cp_name = "checkpoint_nsga.pkl"
            else:
                cp_name = "checkpoint_nsga_{}.pkl".format(id)
            pickle.dump(cp, open(cp_name, "wb"))

        # check hard time limit
        gen_time = datetime.datetime.now() - start_time
        total_time = total_time + gen_time
        # print("Time ", total_time)
        if total_time > datetime.timedelta(hours=4*24):
            print("Time limit exceeded.")
            break

    return population, logbook
