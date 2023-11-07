from bp import *
from GA_optimiser import walk_forward_analysis
import random
import numpy as np
from deap import base, creator, tools, algorithms
from statistics import variance
from multiprocessing import Pool
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def error_check(params,n):
    if len(params) != n:
        raise IndexError('The number of parameters is not correct')
    
import random
import numpy as np
from deap import base, creator, tools, algorithms
from statistics import variance
from multiprocessing import Pool


# Set up the environment for the genetic algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_G", random.uniform, 0, 1)
toolbox.register("attr_n", random.uniform, 0, 1)

# Structure initializers
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_G, toolbox.attr_n), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Genetic operators
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Evaluation function
def evalStrategy(individual):
    G, n = individual
    max_loss, R_PNL, profit, _ = run_strategy_optimised(train_data, G, n)
    return profit,

toolbox.register("evaluate", evalStrategy)

# Constants for the genetic algorithm
NGEN = 40  # Number of generations
MU = 50  # Number of individuals to select for the next generation
LAMBDA = 100  # Number of children to produce at each generation
CXPB = 0.7  # The probability with which two individuals are crossed
MUTPB = 0.2  # Mutation probability
NUM_RUNS = 4  # Number of parallel runs

# Function to run a single GA process
def run_ga(seed):
    random.seed(seed)
    population = toolbox.population(n=MU)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaMuPlusLambda(population, toolbox, mu=MU, lambda_=LAMBDA, 
                              cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, 
                              stats=stats, halloffame=hof, verbose=True)
    print("Best individual is ", hof[0], hof[0].fitness.values[0])
    return hof[0]

# Main optimisation function
def deap_optimiser_g_n(train_data, test_data, parameters, optimization_params):
    """
    Optimizes the parameters of a trading strategy using a genetic algorithm.
    
    Args:
    train_data (pandas.DataFrame): The training data used to optimize the strategy.
    test_data (pandas.DataFrame): The test data used to evaluate the optimized strategy.
    parameters (list): A list of two lists, where the first list contains the grid parameters and the second list contains the position parameters.
    optimization_params (list): A list of three parameters: the number of generations, the number of population, and the maximum number of stagnant generations before early stopping.
    
    Returns:
    tuple: A tuple containing the maximum loss, the return per unit of risk, the profit, and the optimal values of G and n.
    """
    # Extract optimization parameters
    ngen, npop, max_stagnant_gen = optimization_params

    # Error check your parameters (assuming this is a custom function you've defined)
    error_check(parameters, 2)
    
    # Extract the grid and position parameters
    grid_params, position_params = parameters

    # Update the attribute generators according to the parameter ranges
    toolbox.register("attr_G", random.uniform, grid_params[0], grid_params[1])
    toolbox.register("attr_n", random.uniform, position_params[0], position_params[1])

    # If you want to run parallel GA runs, set up a multiprocessing pool
    if __name__ == "__main__":
        try:
            pool = Pool(processes=NUM_RUNS)  # Number of parallel processes
            seeds = [random.randint(1, 10000) for _ in range(NUM_RUNS)]  # Generate random seeds
            results = pool.map(run_ga, seeds)

            # Find the best result from all runs
            best_individual = max(results, key=lambda ind: toolbox.evaluate(ind))
            print("Best individual across all runs is ", best_individual, toolbox.evaluate(best_individual))

            # Evaluate the best individual on the test data
            max_loss, R_PNL, profit, _ = run_strategy_optimised(test_data, *best_individual)
            test_values = [max_loss, R_PNL, profit]
            print("Test values are ", test_values)

            # Evaluate the best individual on the train data
            max_loss, R_PNL, profit, _ = run_strategy_optimised(train_data, *best_individual)
            train_values = [max_loss, R_PNL, profit]
            print("Train values are ", train_values)

            # Before returning, log the values
            logging.info(f"Returning from deap_optimiser_g_n: {test_values}, {train_values}, {best_individual}")
            return test_values, train_values, best_individual
        except Exception as e:
            logging.exception("An error occurred in deap_optimiser_g_n.")
            raise e  # Re-raise the exception to handle it at a higher level
    
try:
    grid_params = [0.001,0.01,0.0005]
    lot_params = [100000,2000000,100000]

    n_grid_params = ((grid_params[1]-grid_params[0])/grid_params[2])
    n_lot_params = ((lot_params[1]-lot_params[0])/lot_params[2])

    print('number of grid params:-',(n_grid_params))
    print('number of lot params:-',(n_lot_params))
    print('total_number_of_combinations:-',(n_grid_params*n_lot_params))

    # Adjust these parameter according to search space
    n_trials = 50 #NGEN
    npop = 100
    early_stopping_gen = n_trials # no early stopping
    optimizer_param = [n_trials, npop, early_stopping_gen]

    print('search_space_explored:-',(n_trials/(n_grid_params*n_lot_params))*100,'%')

    parameters = [grid_params,lot_params]

    results = walk_forward_analysis('jan 2021','jan 2022',1,parameters,optimization_function=deap_optimiser_g_n,optimizer_params=optimizer_param,lookback_in_months=6,evaluation_period=3)
    results
except Exception as e:
    logging.exception("An error occurred in deap_optimiser_g_n.")
    raise e