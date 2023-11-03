__Author__ = "samanvayms"
'''
Genetic Algorithm optimisers using DEAP modified from code written by napaton prasertthum
'''

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from bp import data_gather_from_files,run_strategy_optimised,run_strategy_eval
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
import random
from deap import base, creator, tools

def error_check(params,n):
    if len(params) != n:
        raise IndexError('The number of parameters is not correct')

def generate_date_ranges_for_walk_forward(start_month_year, end_month_year, day=15, n_months = 1):
    # Initialize an empty list to store the date ranges
    date_ranges = []
    if day not in range(1, 29):
        raise ValueError('Day must be between 1 and 28')
    # Convert the input strings to datetime objects, using the given day
    start_date = datetime.strptime(f"{day} {start_month_year}", '%d %b %Y')
    end_date = datetime.strptime(f"{day} {end_month_year}", '%d %b %Y')
    
    # Generate the date ranges
    current_date = start_date
    while current_date <= end_date:
        next_date = current_date + relativedelta(months=n_months)
        date_range = [current_date.strftime('%d %b %Y'), (next_date - relativedelta(days=1)).strftime('%d %b %Y')]
        date_ranges.append(date_range)
        current_date = next_date
    
    return date_ranges[:-1]

def get_previous_n_months(end_date_str, n_months):
    # Convert the input string to a datetime object
    end_date = datetime.strptime(end_date_str, '%d %b %Y')
    
    # Calculate the start date
    start_date = end_date - relativedelta(months=n_months)
    
    # Create the date range
    date_range = [start_date.strftime('%d %b %Y'), (end_date - relativedelta(days=1)).strftime('%d %b %Y')]
    
    return date_range

def walk_forward_analysis(evaluation_start, evaluation_end, evaluation_day,parameters,optimization_function = None, optimizer_params =[],  lookback_in_months = 6,evaluation_period = 3):
    generated_date_ranges = generate_date_ranges_for_walk_forward(evaluation_start, evaluation_end,evaluation_day,n_months = evaluation_period)
    df = {}
    for dates in generated_date_ranges:
        train_period = get_previous_n_months(dates[0], lookback_in_months)
        train_data = data_gather_from_files(train_period[0],train_period[1])['EURUSD.mid']
        print('Data gathered for training period: ',train_period[0],train_period[1])
        test_data = data_gather_from_files(dates[0],dates[1])['EURUSD.mid']
        print('Data gathered for testing period: ',dates[0],dates[1])
        max_loss,R_PNL,profit,optimal_params = optimization_function(train_data,test_data,parameters,optimizer_params)
        print('Optimal parameters are: ',optimal_params)
        print('Max loss,R_PNL,profit are: ',max_loss, R_PNL,profit)
        df[dates[0] +'-'+ dates[1]] = [max_loss, R_PNL,profit]
    df = pd.DataFrame(df).T
    df.columns = ['max_loss', 'R_PNL','profit']
    return df

import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)

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
    ngen = optimization_params[0]  # number of generations
    npop = optimization_params[1]  # number of population

    error_check(parameters,2)
    
    grid_params = parameters[0]
    position_params = parameters[1]

    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) #maximizing
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Define the genes for our individual
    toolbox.register("G_gene", random.randint, grid_params[0]//grid_params[2], grid_params[1]//grid_params[2])
    toolbox.register("n_gene", random.randint, position_params[0]//position_params[2], position_params[1]//position_params[2])

    # Create an individual with the genes
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.G_gene, toolbox.n_gene), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def objective(individual):
        G, n = individual[0]*grid_params[2], individual[1]*position_params[2]
        max_loss, R_PNL, profit, _ = run_strategy_optimised(train_data, G, n)
    
        constraints = [
            max_loss < -500e3
        ]

        if any(constraints):
            return float('-inf'),  # Return large negative value when constraints are not satisfied
        return profit,

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=[grid_params[0]//grid_params[2], position_params[0]//position_params[2]], 
                     up=[grid_params[1]//grid_params[2], position_params[1]//position_params[2]], indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)   ##############
    toolbox.register("evaluate", objective)

    population = toolbox.population(n=npop)
    CXPB, MUTPB = 0.5, 0.2

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    #initiate early stopping
    stagnant_generations = 0  # Counter for generations without improvement
    MAX_STAGNANT_GEN = optimization_params[2]  # Early stopping criterion: stop if no improvement over x generations
    best_fitness_so_far = float('-inf')  # since we're maximizing
    ##
    for gen in range(ngen):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Mutation
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        fitnesses = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        #early stopping
        current_best_fitness = max(ind.fitness.values[0] for ind in population)

        if current_best_fitness > best_fitness_so_far:
            best_fitness_so_far = current_best_fitness
            stagnant_generations = 0  # Reset counter
        else:
            stagnant_generations += 1

        if stagnant_generations >= MAX_STAGNANT_GEN:
            print(f"Early stopping on generation {gen} due to no improvement.")
            break
        ##
        population[:] = offspring

    best_ind = tools.selBest(population, 1)[0]
    optimal_g = np.round(best_ind[0]*grid_params[2],5)
    optimal_n = best_ind[1]*position_params[2]
    print("optimisation completed")
    max_loss, R_PNL,profit,_  = run_strategy_optimised(test_data, optimal_g,optimal_n)
    return max_loss, R_PNL,profit,[optimal_g,optimal_n]

def deap_optimiser_multiplier(train_data, test_data, parameters, optimization_params):
    ngen = optimization_params[0]  # number of generations
    npop = optimization_params[1]  # number of population

    error_check(parameters,4)
    
    grid_params = parameters[0]
    position_params = parameters[1]
    multiplier_params = parameters[2]
    lookback_params = parameters[3]

    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) #maximizing
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Define the genes for our individual
    toolbox.register("G_gene", random.randint, grid_params[0]//grid_params[2], grid_params[1]//grid_params[2])
    toolbox.register("n_gene", random.randint, position_params[0]//position_params[2], position_params[1]//position_params[2])

    toolbox.register("multiplier_gene", random.randint, multiplier_params[0]//multiplier_params[2], multiplier_params[1]//multiplier_params[2])
    toolbox.register("lookback_gene", random.choice, lookback_params)

    CXPB, MUTPB = 0.5, 0.2

    def custom_mutate(individual): 
        if random.random() < MUTPB:
            individual[0] = random.randint(grid_params[0]//grid_params[2], grid_params[1]//grid_params[2])
        if random.random() < MUTPB:
            individual[1] = random.randint(position_params[0]//position_params[2], position_params[1]//position_params[2])
        if random.random() < MUTPB:
            individual[2] = random.randint(multiplier_params[0]//multiplier_params[2], multiplier_params[1]//multiplier_params[2])
        if random.random() < MUTPB:
            individual[3] = random.choice(lookback_params)
        return individual,

    toolbox.register("mutate", custom_mutate)

    # Create an individual with the genes
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.G_gene, toolbox.n_gene, toolbox.multiplier_gene, toolbox.lookback_gene), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def objective(individual):
        G, n, multiplier, lookback = individual[0]*grid_params[2], individual[1]*position_params[2], individual[2]*multiplier_params[2], individual[3]

        max_loss, R_PNL, profit, _ = run_strategy_optimised(train_data, G, n, multiplier = multiplier,lookback = lookback)
    
        constraints = [
            max_loss < -500e3
        ]

        if any(constraints):
            return float('-inf'),  # Return large negative value when constraints are not satisfied
        return profit,

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize=3)   
    toolbox.register("evaluate", objective)

    population = toolbox.population(n=npop)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    #initiate early stopping
    stagnant_generations = 0  # Counter for generations without improvement
    MAX_STAGNANT_GEN = optimization_params[2]  # Early stopping criterion: stop if no improvement over x generations
    best_fitness_so_far = float('-inf')  # since we're maximizing
    ##
    for gen in range(ngen):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Mutation
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        fitnesses = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        #early stopping
        current_best_fitness = max(ind.fitness.values[0] for ind in population)

        if current_best_fitness > best_fitness_so_far:
            best_fitness_so_far = current_best_fitness
            stagnant_generations = 0  # Reset counter
        else:
            stagnant_generations += 1

        if stagnant_generations >= MAX_STAGNANT_GEN:
            print(f"Early stopping on generation {gen} due to no improvement.")
            break
        ##
        population[:] = offspring

    best_ind = tools.selBest(population, 1)[0]
    optimal_g = np.round(best_ind[0]*grid_params[2],5)
    optimal_n = best_ind[1]*position_params[2]
    optimal_m = np.round(best_ind[2] *multiplier_params[2],5)
    optimal_l = best_ind[3]

    print("optimisation completed")
    max_loss, R_PNL,profit,_  = run_strategy_optimised(test_data, optimal_g,optimal_n,multiplier = optimal_m,lookback = optimal_l)
    return max_loss, R_PNL,profit,[optimal_g,optimal_n,optimal_m,optimal_l]

# **********************************************************************************************************************************************************

def get_date_pairs_(start_date_str,end_date_str,date_format='%d %b %Y',interval = 1):
    # Parse the start and end dates
    start_date = datetime.strptime(start_date_str, date_format)
    end_date = datetime.strptime(end_date_str, date_format)

    # Initialize an empty list to hold the interval pairs
    interval_list = []

    # Increment by one month until we reach or surpass the end date
    current_date = start_date
    next_date = current_date + relativedelta(months=interval)
    while next_date <= end_date:
        interval_list.append((current_date.strftime(date_format), next_date.strftime(date_format)))
        current_date = next_date
        next_date += relativedelta(months=interval)

    # Add the remaining interval if there are extra days left
    if current_date != end_date:
        interval_list.append((current_date.strftime(date_format), end_date.strftime(date_format)))
        
    return interval_list

def walk_forward_analysis_rollover(evaluation_start, evaluation_end, evaluation_day,parameters,optimization_function, optimizer_params =[],  lookback_in_months = 6,evaluation_period = 3,position_turnover = 1):
    generated_date_ranges = generate_date_ranges_for_walk_forward(evaluation_start, evaluation_end,evaluation_day,n_months = evaluation_period)
    df = {}
    train_dfs = []
    test_dfs = []
    for dates in generated_date_ranges:
        train_period = get_previous_n_months(dates[0], lookback_in_months)
        print('Data gathered for testing period: ',dates[0],dates[1])
        optimal_params,train_df,test_df = optimization_function(train_period[0],train_period[1],dates[0],dates[1],parameters, optimizer_params, position_turnover)
        print('Optimal parameters are: ',optimal_params)
        max_loss = min(np.min(np.cumsum(test_df['profit'])),np.min(test_df['max_loss']))
        profit = np.sum(test_df['profit'])
        print('Max loss,profit are: ',max_loss, profit)
        df[dates[0] +'-'+ dates[1]] = [max_loss, profit]
        train_dfs.append(train_df)
        test_dfs.append(test_df)
    df = pd.DataFrame(df).T
    df.columns = ['max_loss','profit']
    return df,train_dfs,test_dfs

def deap_optimiser_g_n_rollover(train_start,train_end,test_start,test_end,parameters, optimization_params, position_turnover=1):
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
    train_pairs = get_date_pairs_(train_start, train_end, interval = position_turnover)
    test_pairs = get_date_pairs_(test_start, test_end, interval = position_turnover)
    
    ngen = optimization_params[0]  # number of generations
    npop = optimization_params[1]  # number of population

    error_check(parameters,2)
    
    grid_params = parameters[0]
    position_params = parameters[1]

    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) #maximizing
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Define the genes for our individual
    toolbox.register("G_gene", random.randint, grid_params[0]//grid_params[2], grid_params[1]//grid_params[2])
    toolbox.register("n_gene", random.randint, position_params[0]//position_params[2], position_params[1]//position_params[2])

    # Create an individual with the genes
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.G_gene, toolbox.n_gene), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def objective(individual):
        G, n = individual[0]*grid_params[2], individual[1]*position_params[2]
        profit = 0
        for pair in train_pairs:
            train_data = data_gather_from_files(pair[0],pair[1])[EURUSD.mid]
            max_loss, R_PNL, month_profit, _ = run_strategy_optimised(train_data, G, n)
            constraints = [
                max_loss < -500e3
            ]
            if any(constraints):
                profit += -np.inf # Return large negative value when constraints are not satisfied
            else:
                profit += month_profit
        return profit,

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=[grid_params[0]//grid_params[2], position_params[0]//position_params[2]], 
                     up=[grid_params[1]//grid_params[2], position_params[1]//position_params[2]], indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)   ##############
    toolbox.register("evaluate", objective)

    population = toolbox.population(n=npop)
    CXPB, MUTPB = 0.5, 0.2

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    #initiate early stopping
    stagnant_generations = 0  # Counter for generations without improvement
    MAX_STAGNANT_GEN = optimization_params[2]  # Early stopping criterion: stop if no improvement over x generations
    best_fitness_so_far = float('-inf')  # since we're maximizing
    ##
    for gen in range(ngen):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Mutation
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        fitnesses = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        #early stopping
        current_best_fitness = max(ind.fitness.values[0] for ind in population)

        if current_best_fitness > best_fitness_so_far:
            best_fitness_so_far = current_best_fitness
            stagnant_generations = 0  # Reset counter
        else:
            stagnant_generations += 1

        if stagnant_generations >= MAX_STAGNANT_GEN:
            print(f"Early stopping on generation {gen} due to no improvement.")
            break
        ##
        population[:] = offspring

    best_ind = tools.selBest(population, 1)[0]
    optimal_g = np.round(best_ind[0]*grid_params[2],5)
    optimal_n = best_ind[1]*position_params[2]
    
    train_df = {}
    test_df = {}
    
    for pair in train_pairs:
        tick_data = data_gather_from_files(pair[0],pair[1])['EURUSD.mid']
        max_loss, R_PNL,profit, _ = run_strategy_optimised(tick_data, optimal_g,optimal_n)
        train_df[pair[0] +'-'+ pair[1]] = [max_loss, R_PNL,profit,optimal_g,optimal_n]
    print("optimisation completed")
    
    for pair in test_pairs:
        tick_data = data_gather_from_files(pair[0],pair[1])['EURUSD.mid']
        max_loss, R_PNL,profit, _ = run_strategy_optimised(tick_data, optimal_g,optimal_n)
        test_df[pair[0] +'-'+ pair[1]] = [max_loss, R_PNL,profit,optimal_g,optimal_n]
    
    train_df = pd.DataFrame(train_df).T
    train_df.columns = ['max_loss', 'R_PNL','profit','optimal_g','optimal_n']
    
    test_df = pd.DataFrame(test_df).T
    test_df.columns = ['max_loss', 'R_PNL','profit','optimal_g','optimal_n']
    return [optimal_g,optimal_n],train_df,test_df

def deap_optimiser_multiplier_rollover(train_start,train_end,test_start,test_end,parameters, optimization_params, position_turnover=1):
    train_pairs = get_date_pairs_(train_start, train_end, interval = position_turnover)
    test_pairs = get_date_pairs_(test_start, test_end, interval = position_turnover)
    
    ngen = optimization_params[0]  # number of generations
    npop = optimization_params[1]  # number of population

    error_check(parameters,4)
    
    grid_params = parameters[0]
    position_params = parameters[1]
    multiplier_params = parameters[2]
    lookback_params = parameters[3]

    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) #maximizing
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Define the genes for our individual
    toolbox.register("G_gene", random.randint, grid_params[0]//grid_params[2], grid_params[1]//grid_params[2])
    toolbox.register("n_gene", random.randint, position_params[0]//position_params[2], position_params[1]//position_params[2])

    toolbox.register("multiplier_gene", random.randint, multiplier_params[0]//multiplier_params[2], multiplier_params[1]//multiplier_params[2])
    toolbox.register("lookback_gene", random.choice, lookback_params)

    CXPB, MUTPB = 0.5, 0.2

    def custom_mutate(individual): 
        if random.random() < MUTPB:
            individual[0] = random.randint(grid_params[0]//grid_params[2], grid_params[1]//grid_params[2])
        if random.random() < MUTPB:
            individual[1] = random.randint(position_params[0]//position_params[2], position_params[1]//position_params[2])
        if random.random() < MUTPB:
            individual[2] = random.randint(multiplier_params[0]//multiplier_params[2], multiplier_params[1]//multiplier_params[2])
        if random.random() < MUTPB:
            individual[3] = random.choice(lookback_params)
        return individual,

    toolbox.register("mutate", custom_mutate)

    # Create an individual with the genes
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.G_gene, toolbox.n_gene, toolbox.multiplier_gene, toolbox.lookback_gene), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def objective(individual):
        G, n, multiplier, lookback = individual[0]*grid_params[2], individual[1]*position_params[2], individual[2]*multiplier_params[2], individual[3]
        profit = 0
        for pair in train_pairs:
            train_data = data_gather_from_files(pair[0],pair[1])['EURUSD.mid']
            max_loss, R_PNL, month_profit, _ = run_strategy_optimised(train_data, G, n, multiplier = multiplier,lookback = lookback)
            constraints = [
                max_loss < -500e3
            ]
            if any(constraints):
                profit += -np.inf # Return large negative value when constraints are not satisfied
            else:
                profit += month_profit
        return profit,
    
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize=3)   
    toolbox.register("evaluate", objective)

    population = toolbox.population(n=npop)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    #initiate early stopping
    stagnant_generations = 0  # Counter for generations without improvement
    MAX_STAGNANT_GEN = optimization_params[2]  # Early stopping criterion: stop if no improvement over x generations
    best_fitness_so_far = float('-inf')  # since we're maximizing
    ##
    for gen in range(ngen):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Mutation
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        fitnesses = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        #early stopping
        current_best_fitness = max(ind.fitness.values[0] for ind in population)

        if current_best_fitness > best_fitness_so_far:
            best_fitness_so_far = current_best_fitness
            stagnant_generations = 0  # Reset counter
        else:
            stagnant_generations += 1

        if stagnant_generations >= MAX_STAGNANT_GEN:
            print(f"Early stopping on generation {gen} due to no improvement.")
            break
        ##
        population[:] = offspring

    best_ind = tools.selBest(population, 1)[0]
    optimal_g = np.round(best_ind[0]*grid_params[2],5)
    optimal_n = best_ind[1]*position_params[2]
    optimal_m = np.round(best_ind[2] *multiplier_params[2],5)
    optimal_l = best_ind[3]

    train_df = {}
    test_df = {}
    
    for pair in train_pairs:
        tick_data = data_gather_from_files(pair[0],pair[1])['EURUSD.mid']
        max_loss, R_PNL,profit, _ = run_strategy_optimised(tick_data, optimal_g,optimal_n,multiplier = optimal_m,lookback = optimal_l)
        train_df[pair[0] +'-'+ pair[1]] = [max_loss, R_PNL,profit,optimal_g,optimal_n,optimal_m,optimal_l]
    print("optimisation completed")
    
    for pair in test_pairs:
        tick_data = data_gather_from_files(pair[0],pair[1])['EURUSD.mid']
        max_loss, R_PNL,profit, _ = run_strategy_optimised(tick_data, optimal_g,optimal_n,multiplier = optimal_m,lookback = optimal_l)
        test_df[pair[0] +'-'+ pair[1]] = [max_loss, R_PNL,profit,optimal_g,optimal_n,optimal_m,optimal_l]
    
    train_df = pd.DataFrame(train_df).T
    train_df.columns = ['max_loss', 'R_PNL','profit','optimal_g','optimal_n','optimal_m','optimal_l']
    
    test_df = pd.DataFrame(test_df).T
    test_df.columns = ['max_loss', 'R_PNL','profit','optimal_g','optimal_n','optimal_m','optimal_l']
    
    return [optimal_g,optimal_n,optimal_m,optimal_l],train_df,test_df