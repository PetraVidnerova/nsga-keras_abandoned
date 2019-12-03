import click
import pickle
import matplotlib.pyplot as plt

from deap import creator, base

from individual import Individual
from convindividual import ConvIndividual
from dataset import load_data
from config import Config
from utils import error, print_stat

USE_CONV = False


def init():
    creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual",
                   ConvIndividual if USE_CONV else Individual,
                   fitness=creator.FitnessMax)


def load_checkpoint(name):
    init()
    cp = pickle.load(open(name, "rb"))
    # print(cp.keys())
    pop = cp["population"]
    front = cp["halloffame"]
    log = cp["logbook"]
    return pop, front, log


@click.group()
@click.option("--conv", default=False, type=bool)
def main(conv):
    global USE_CONV

    if conv:
        USE_CONV = True


@main.command()
@click.argument("cp_name")
def show_pop(cp_name):
    pop, _, _ = load_checkpoint(cp_name)
    print(len(pop))
    print(" i: acc    size")
    for i, ind in enumerate(pop):
        print(f"{i:2}: {ind.fitness.values[0]*100:.2f} ",
              f"{ind.fitness.values[1]:5.1f}")


@main.command()
@click.argument("i", type=int)
@click.argument("trainset")
@click.argument("testset")
@click.argument("cp_name")
def evaluate(i, trainset, testset, cp_name):
    pop, _, _ = load_checkpoint(cp_name)

    # load the whole data
    X_train, y_train = load_data("data/" + trainset)
    X_test, y_test = load_data("data/" + testset)

    E_train, E_test = [], []  # list of accuracies
    for _ in range(5):
        network = pop[i].createNetwork()
        network.fit(X_train, y_train,
                    batch_size=Config.batch_size,
                    nb_epoch=20,
                    verbose=0)

        yy_train = network.predict(X_train)
        E_train.append(error(yy_train, y_train))

        yy_test = network.predict(X_test)
        E_test.append(error(yy_test, y_test))

    print_stat(E_train, "train")
    print_stat(E_test, "test")
    print(pop[i].fitness.values)


@main.command()
@click.argument("cp_name")
def plot(cp_name):
    _, _, log = load_checkpoint(cp_name)
    acc_avg = [line["avg"][0] for line in log]
    acc_size = [line["avg"][1] for line in log]

    acc_max = [line["max"][0] for line in log]
    size_min = [line["min"][1] for line in log]

    ax1, ax2 = plt.subplot(221), plt.subplot(222)

    ax1.plot(acc_avg, color="blue")
    ax2.plot(acc_size, color="green")
    ax1.set_title("avg acc")
    ax2.set_title("avg size")

    ax3, ax4 = plt.subplot(223), plt.subplot(224)
    ax3.plot(acc_max, color="blue")
    ax4.plot(size_min, color="green")
    ax3.set_title("max acc")
    ax4.set_title("min size")

    plt.show()


if __name__ == "__main__":
    main()
