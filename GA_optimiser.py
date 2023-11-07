__Author__ = "samanvayms"
'''
Genetic Algorithm optimisers using DEAP with contributions from Napaton 'Jenny' Prasertthum
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
import os

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

def walk_forward_analysis(evaluation_start, evaluation_end, evaluation_day,parameters,optimization_function, optimizer_params =[],  lookback_in_months = 6,evaluation_period = 3,currency_pair='EURUSD'):
    generated_date_ranges = generate_date_ranges_for_walk_forward(evaluation_start, evaluation_end,evaluation_day,n_months = evaluation_period)
    train_df = {}
    test_df = {}
    optimal_df = {}
    for dates in generated_date_ranges:
        train_period = get_previous_n_months(dates[0], lookback_in_months)
        train_data = data_gather_from_files(train_period[0],train_period[1],currency_pair)['mid']
        print('Data gathered for training period: ',train_period[0],train_period[1])
        test_data = data_gather_from_files(dates[0],dates[1])['mid']
        print('Data gathered for testing period: ',dates[0],dates[1])
        test_values,train_values,optimal_params = optimization_function(train_data,test_data,parameters,optimizer_params)
        print('Optimal parameters are: ',optimal_params)
        test_df[dates[0] +'-'+ dates[1]] = test_values
        train_df[train_period[0] +'-'+ train_period[1]] = train_values
        optimal_df[dates[0] +'-'+ dates[1]] = optimal_params
    test_df = pd.DataFrame(test_df).T
    test_df.columns = ['max_loss','R_PNL','profit']
    train_df = pd.DataFrame(train_df).T
    train_df.columns = ['max_loss','R_PNL','profit']
    optimal_df = pd.DataFrame(optimal_df).T
    if len(parameters) == 2:
        optimal_df.columns = ['optimal_g','optimal_n']
    elif len(parameters) == 4:
        optimal_df.columns = ['optimal_g','optimal_n','optimal_m','optimal_l']
    elif len(parameters) == 5:
        optimal_df.columns = ['optimal_g','optimal_n','optimal_i','optimal_l','optimal_s']
    else:
        pass
    return test_df,train_df,optimal_df

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

    random.seed(42)
        
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
    test_values = [max_loss, R_PNL,profit]
    max_loss, R_PNL,profit,_  = run_strategy_optimised(train_data, optimal_g,optimal_n)
    train_values = [max_loss, R_PNL,profit]
    optimal_values = [optimal_g,optimal_n]
    return test_values,train_values,optimal_values
    

def deap_optimiser_multiplier(train_data, test_data, parameters, optimization_params):
    ngen = optimization_params[0]  # number of generations
    npop = optimization_params[1]  # number of population

    error_check(parameters,4)
    
    grid_params = parameters[0]
    position_params = parameters[1]
    multiplier_params = parameters[2]
    lookback_params = parameters[3]

    random.seed(42)
    
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
    max_loss, R_PNL,profit,_  = run_strategy_optimised(test_data, optimal_g,optimal_n, multiplier = optimal_m,lookback = optimal_l)
    test_values = [max_loss, R_PNL,profit]
    max_loss, R_PNL,profit,_  = run_strategy_optimised(train_data, optimal_g,optimal_n, multiplier = optimal_m,lookback = optimal_l)
    train_values = [max_loss, R_PNL,profit]
    optimal_values = [optimal_g,optimal_n,optimal_m,optimal_l]
    return test_values,train_values,optimal_values

def deap_optimiser_indicator(train_data, test_data, parameters, optimization_params):
    ngen = optimization_params[0]  # number of generations
    npop = optimization_params[1]  # number of population

    error_check(parameters,5)

    grid_params = parameters[0]
    position_params = parameters[1]
    indicator_type_params = parameters[2]
    lookback_params = parameters[3]
    scaling_factor_params = parameters[4]

    random.seed(42)
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) #maximizing
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Define the genes for our individual
    toolbox.register("G_gene", random.randint, grid_params[0]//grid_params[2], grid_params[1]//grid_params[2])
    toolbox.register("n_gene", random.randint, position_params[0]//position_params[2], position_params[1]//position_params[2])

    toolbox.register("indicator_type_gene", random.choice, indicator_type_params)
    toolbox.register("lookback_gene", random.choice, lookback_params)
    toolbox.register("scaling_factor_gene", random.randint, scaling_factor_params[0]//scaling_factor_params[2], scaling_factor_params[1]//scaling_factor_params[2])

    CXPB, MUTPB = 0.5, 0.2

    def custom_mutate(individual): 
        if random.random() < MUTPB:
            individual[0] = random.randint(grid_params[0]//grid_params[2], grid_params[1]//grid_params[2])
        if random.random() < MUTPB:
            individual[1] = random.randint(position_params[0]//position_params[2], position_params[1]//position_params[2])
        if random.random() < MUTPB:
            individual[2] = random.choice(indicator_type_params)
        if random.random() < MUTPB:
            individual[3] = random.choice(lookback_params)
        if random.random() < MUTPB:
            individual[4] = random.randint(scaling_factor_params[0]//scaling_factor_params[2], scaling_factor_params[1]//scaling_factor_params[2])
        return individual,

    toolbox.register("mutate", custom_mutate)

    # Create an individual with the genes
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.G_gene, toolbox.n_gene, toolbox.indicator_type_gene, toolbox.lookback_gene, toolbox.scaling_factor_gene), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def objective(individual):
        G, n, indicator_type, lookback, scaling_factor = individual[0]*grid_params[2], individual[1]*position_params[2], individual[2], individual[3], individual[4]*scaling_factor_params[2]

        max_loss, R_PNL, profit, _ = run_strategy_optimised(train_data, G, n, indicator_type = indicator_type, lookback = lookback, indicator_scale = scaling_factor)
    
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
    optimal_i = best_ind[2]
    optimal_l = best_ind[3]
    optimal_s = best_ind[4]*scaling_factor_params[2]

    print("optimisation completed")
    max_loss, R_PNL,profit,_  = run_strategy_optimised(test_data, optimal_g,optimal_n, indicator_type = optimal_i, lookback = optimal_l, indicator_scale = optimal_s)
    test_values = [max_loss, R_PNL,profit]
    max_loss, R_PNL,profit,_  = run_strategy_optimised(train_data, optimal_g,optimal_n, indicator_type = optimal_i, lookback = optimal_l, indicator_scale = optimal_s)
    train_values = [max_loss, R_PNL,profit]
    optimal_values = [optimal_g,optimal_n,optimal_i,optimal_l,optimal_s]
    return test_values,train_values,optimal_values

def deap_optimiser_indicator_std(train_data, test_data, parameters, optimization_params):
    ngen = optimization_params[0]  # number of generations
    npop = optimization_params[1]  # number of population

    error_check(parameters,5)

    grid_params = parameters[0]
    position_params = parameters[1]
    indicator_type_params = parameters[2]
    lookback_params = parameters[3]
    scaling_factor_params = parameters[4]

    random.seed(42)
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) #maximizing
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Define the genes for our individual
    toolbox.register("G_gene", random.randint, grid_params[0]//grid_params[2], grid_params[1]//grid_params[2])
    toolbox.register("n_gene", random.randint, position_params[0]//position_params[2], position_params[1]//position_params[2])

    toolbox.register("indicator_type_gene", random.choice, indicator_type_params)
    toolbox.register("lookback_gene", random.choice, lookback_params)
    toolbox.register("scaling_factor_gene", random.randint, scaling_factor_params[0]//scaling_factor_params[2], scaling_factor_params[1]//scaling_factor_params[2])

    CXPB, MUTPB = 0.5, 0.2

    def custom_mutate(individual): 
        if random.random() < MUTPB:
            individual[0] = random.randint(grid_params[0]//grid_params[2], grid_params[1]//grid_params[2])
        if random.random() < MUTPB:
            individual[1] = random.randint(position_params[0]//position_params[2], position_params[1]//position_params[2])
        if random.random() < MUTPB:
            individual[2] = random.choice(indicator_type_params)
        if random.random() < MUTPB:
            individual[3] = random.choice(lookback_params)
        if random.random() < MUTPB:
            individual[4] = random.randint(scaling_factor_params[0]//scaling_factor_params[2], scaling_factor_params[1]//scaling_factor_params[2])
        return individual,

    toolbox.register("mutate", custom_mutate)

    # Create an individual with the genes
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.G_gene, toolbox.n_gene, toolbox.indicator_type_gene, toolbox.lookback_gene, toolbox.scaling_factor_gene), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def objective(individual):
        G, n, indicator_type, lookback, scaling_factor = individual[0]*grid_params[2], individual[1]*position_params[2], individual[2], individual[3], individual[4]*scaling_factor_params[2]

        max_loss, R_PNL, profit, std = run_strategy_optimised(train_data, G, n, indicator_type = indicator_type, lookback = lookback, indicator_scale = scaling_factor)
    
        constraints = [
            max_loss < -500e3
        ]

        if any(constraints):
            return float('-inf'),  # Return large negative value when constraints are not satisfied
        return (profit/std)*individual[1],
    
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
    optimal_i = best_ind[2]
    optimal_l = best_ind[3]
    optimal_s = best_ind[4]*scaling_factor_params[2]

    print("optimisation completed")
    max_loss, R_PNL,profit,_  = run_strategy_optimised(test_data, optimal_g,optimal_n, indicator_type = optimal_i, lookback = optimal_l, indicator_scale = optimal_s)
    test_values = [max_loss, R_PNL,profit]
    max_loss, R_PNL,profit,_  = run_strategy_optimised(train_data, optimal_g,optimal_n, indicator_type = optimal_i, lookback = optimal_l, indicator_scale = optimal_s)
    train_values = [max_loss, R_PNL,profit]
    optimal_values = [optimal_g,optimal_n,optimal_i,optimal_l,optimal_s]
    return test_values,train_values,optimal_values

def gen_results(start,end,train_length,test_length,currency_pair='EURUSD',optimiser=deap_optimiser_indicator,parameters=[],optimizer_param=[]):
    # Define the folder path and the CSV file name
    if optimiser == deap_optimiser_indicator:
        folder_path = 'results/'+currency_pair+ '/profit'
    elif optimiser == deap_optimiser_indicator_std:
        folder_path = 'results/'+currency_pair+ '/std'
    else:
        return 'Invalid optimiser'
    csv_file_name = f'{train_length}-{test_length}-{start}-{end}'    # Replace with your actual CSV file name
    full_path = os.path.join(folder_path, csv_file_name)

    # Check if the folder exists
    if not os.path.exists(folder_path):
        # If the folder does not exist, create it
        os.makedirs(folder_path)

    test_df,train_df,optimal_df = walk_forward_analysis(start,end,1,parameters,optimiser,optimizer_param,train_length,test_length,currency_pair=currency_pair)
    test_path = full_path + '-test.csv'
    train_path = full_path + '-train.csv'
    optimal_df_path = full_path + '-optimal.csv'
    test_df.to_csv(test_path)
    train_df.to_csv(train_path)
    optimal_df.to_csv(optimal_df_path)

# **********************************************************************************************************************************************************

def get_date_pairs_(start_date_str,end_date_str,date_format='%d %b %Y',interval = 1):
    """
    This function generates a list of date pairs given a start date and an end date. The dates are parsed using the specified date format. The interval parameter specifies the number of months between each date pair. If there are extra days left, the remaining interval is added to the last date pair.

    Args:
    - start_date_str (str): The start date in string format.
    - end_date_str (str): The end date in string format.
    - date_format (str): The format of the dates in the input strings. Default is '%d %b %Y'.
    - interval (int): The number of months between each date pair. Default is 1.

    Returns:
    - interval_list (list): A list of tuples containing the date pairs.
    """
    
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
        print('training period ',train_period[0],train_period[1],'test period',dates[0],dates[1])
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

def deap_optimiser_g_n_rollover(train_start,train_end,test_start,test_end,parameters, optimization_params, position_turnover=1,currency_pair='EURUSD'):
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

    random.seed(42)
    
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
            train_data = data_gather_from_files(pair[0],pair[1],currency_pair)['mid']
            max_loss, R_PNL, month_profit, _ = run_strategy_optimised(train_data, G, n)
            constraints = [
                max_loss < -500e3
            ]
            if any(constraints):
                profit += -np.inf # Return large negative value when constraints are not satisfied
            else:
                profit += month_profit
        print('iteration completed')
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
        tick_data = data_gather_from_files(pair[0],pair[1],currency_pair)['mid']
        max_loss, R_PNL,profit, _ = run_strategy_optimised(tick_data, optimal_g,optimal_n)
        train_df[pair[0] +'-'+ pair[1]] = [max_loss, R_PNL,profit,optimal_g,optimal_n]
    print("optimisation completed")
    
    for pair in test_pairs:
        tick_data = data_gather_from_files(pair[0],pair[1],currency_pair)['mid']
        max_loss, R_PNL,profit, _ = run_strategy_optimised(tick_data, optimal_g,optimal_n)
        test_df[pair[0] +'-'+ pair[1]] = [max_loss, R_PNL,profit,optimal_g,optimal_n]
    
    train_df = pd.DataFrame(train_df).T
    train_df.columns = ['max_loss', 'R_PNL','profit','optimal_g','optimal_n']
    
    test_df = pd.DataFrame(test_df).T
    test_df.columns = ['max_loss', 'R_PNL','profit','optimal_g','optimal_n']
    return [optimal_g,optimal_n],train_df,test_df

def deap_optimiser_multiplier_rollover(train_start,train_end,test_start,test_end,parameters, optimization_params, position_turnover=1,currency_pair='EURUSD'):
    train_pairs = get_date_pairs_(train_start, train_end, interval = position_turnover)
    test_pairs = get_date_pairs_(test_start, test_end, interval = position_turnover)
    
    ngen = optimization_params[0]  # number of generations
    npop = optimization_params[1]  # number of population

    error_check(parameters,4)
    
    grid_params = parameters[0]
    position_params = parameters[1]
    multiplier_params = parameters[2]
    lookback_params = parameters[3]

    random.seed(42)
    
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
            train_data = data_gather_from_files(pair[0],pair[1],currency_pair)['mid']
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
        tick_data = data_gather_from_files(pair[0],pair[1],currency_pair)['mid']
        max_loss, R_PNL,profit, _ = run_strategy_optimised(tick_data, optimal_g,optimal_n,multiplier = optimal_m,lookback = optimal_l)
        train_df[pair[0] +'-'+ pair[1]] = [max_loss, R_PNL,profit,optimal_g,optimal_n,optimal_m,optimal_l]
    print("optimisation completed")
    
    for pair in test_pairs:
        tick_data = data_gather_from_files(pair[0],pair[1],currency_pair)['mid']
        max_loss, R_PNL,profit, _ = run_strategy_optimised(tick_data, optimal_g,optimal_n,multiplier = optimal_m,lookback = optimal_l)
        test_df[pair[0] +'-'+ pair[1]] = [max_loss, R_PNL,profit,optimal_g,optimal_n,optimal_m,optimal_l]
    
    train_df = pd.DataFrame(train_df).T
    train_df.columns = ['max_loss', 'R_PNL','profit','optimal_g','optimal_n','optimal_m','optimal_l']
    
    test_df = pd.DataFrame(test_df).T
    test_df.columns = ['max_loss', 'R_PNL','profit','optimal_g','optimal_n','optimal_m','optimal_l']
    
    return [optimal_g,optimal_n,optimal_m,optimal_l],train_df,test_df

def deap_optimiser_indicator_rollover(train_start,train_end,test_start,test_end,parameters, optimization_params, position_turnover=1,currency_pair='EURUSD'):
    train_pairs = get_date_pairs_(train_start, train_end, interval = position_turnover)
    test_pairs = get_date_pairs_(test_start, test_end, interval = position_turnover)
    
    ngen = optimization_params[0]  # number of generations
    npop = optimization_params[1]  # number of population

    error_check(parameters,5)

    grid_params = parameters[0]
    position_params = parameters[1]
    indicator_type_params = parameters[2]
    lookback_params = parameters[3]
    scaling_factor_params = parameters[4]

    random.seed(42)
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) #maximizing
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Define the genes for our individual
    toolbox.register("G_gene", random.randint, grid_params[0]//grid_params[2], grid_params[1]//grid_params[2])
    toolbox.register("n_gene", random.randint, position_params[0]//position_params[2], position_params[1]//position_params[2])

    toolbox.register("indicator_type_gene", random.choice, indicator_type_params)
    toolbox.register("lookback_gene", random.choice, lookback_params)
    toolbox.register("scaling_factor_gene", random.randint, scaling_factor_params[0]//scaling_factor_params[2], scaling_factor_params[1]//scaling_factor_params[2])

    CXPB, MUTPB = 0.5, 0.2

    def custom_mutate(individual): 
        if random.random() < MUTPB:
            individual[0] = random.randint(grid_params[0]//grid_params[2], grid_params[1]//grid_params[2])
        if random.random() < MUTPB:
            individual[1] = random.randint(position_params[0]//position_params[2], position_params[1]//position_params[2])
        if random.random() < MUTPB:
            individual[2] = random.choice(indicator_type_params)
        if random.random() < MUTPB:
            individual[3] = random.choice(lookback_params)
        if random.random() < MUTPB:
            individual[4] = random.randint(scaling_factor_params[0]//scaling_factor_params[2], scaling_factor_params[1]//scaling_factor_params[2])
        return individual,

    toolbox.register("mutate", custom_mutate)

    # Create an individual with the genes
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.G_gene, toolbox.n_gene, toolbox.indicator_type_gene, toolbox.lookback_gene, toolbox.scaling_factor_gene), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def objective(individual):
        G, n, indicator_type, lookback, scaling_factor = individual[0]*grid_params[2], individual[1]*position_params[2], individual[2], individual[3], individual[4]*scaling_factor_params[2]
        profit = 0
        for pair in train_pairs:
            train_data = data_gather_from_files(pair[0],pair[1],currency_pair)['mid']
            max_loss, R_PNL, month_profit, _ = run_strategy_optimised(train_data, G, n,lookback = lookback,indicator_type = indicator_type,indicator_scale=scaling_factor)
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
    optimal_i = best_ind[2]
    optimal_l = best_ind[3]
    optimal_s = best_ind[4]*scaling_factor_params[2]

    train_df = {}
    test_df = {}
    
    for pair in train_pairs:
        tick_data = data_gather_from_files(pair[0],pair[1],currency_pair)['mid']
        max_loss, R_PNL,profit, _ = run_strategy_optimised(tick_data, optimal_g,optimal_n,lookback = optimal_l,indicator_type = optimal_i,indicator_scale=optimal_s)
        train_df[pair[0] +'-'+ pair[1]] = [max_loss, R_PNL,profit,optimal_g,optimal_n,optimal_i,optimal_l,optimal_s]
    print("optimisation completed")
    
    for pair in test_pairs:
        tick_data = data_gather_from_files(pair[0],pair[1],currency_pair)['mid']
        max_loss, R_PNL,profit, _ = run_strategy_optimised(tick_data, optimal_g,optimal_n,lookback = optimal_l,indicator_type = optimal_i,indicator_scale=optimal_s)
        test_df[pair[0] +'-'+ pair[1]] = [max_loss, R_PNL,profit,optimal_g,optimal_n,optimal_i,optimal_l,optimal_s]
    
    train_df = pd.DataFrame(train_df).T
    train_df.columns = ['max_loss', 'R_PNL','profit','optimal_g','optimal_n','optimal_i','optimal_l','optimal_s']
    
    test_df = pd.DataFrame(test_df).T
    test_df.columns = ['max_loss', 'R_PNL','profit','optimal_g','optimal_n','optimal_i','optimal_l','optimal_s']
    
    return [optimal_g,optimal_n,optimal_i,optimal_l,optimal_s],train_df,test_df

def grid_search_optimiser_rollover(train_start,train_end,test_start,test_end,parameters, optimization_params = [], position_turnover=1,currency_pair='EURUSD'):
    train_pairs = get_date_pairs_(train_start, train_end, interval = position_turnover)
    test_pairs = get_date_pairs_(test_start, test_end, interval = position_turnover)
    
    error_check(parameters,2)
    
    grid_params = parameters[0]
    position_params = parameters[1]
    
    ladder_sizing_grid = np.arange(grid_params[0],grid_params[1],grid_params[2])
    lot_sizing_grid = np.arange(position_params[0],position_params[1],position_params[2])
    
    max_profit = -np.inf
    optimal_g = 0
    optimal_n = 0
    

    for ladder_size in ladder_sizing_grid:
        for lot_size in lot_sizing_grid:
            combination_profit = 0
            for pair in train_pairs:
                train_data = data_gather_from_files(pair[0],pair[1],currency_pair)['mid']
                max_loss, R_PNL, profit,_ = run_strategy_optimised(train_data, ladder_size, lot_size)
                if (max_loss > -500000):
                    combination_profit += profit
                else:
                    combination_profit = -np.inf
            if combination_profit > max_profit:
                optimal_g = ladder_size
                optimal_n = lot_size

    train_df = {}
    test_df = {}
    
    for pair in train_pairs:
        tick_data = data_gather_from_files(pair[0],pair[1],currency_pair)['mid']
        max_loss, R_PNL,profit, _ = run_strategy_optimised(tick_data, optimal_g,optimal_n)
        train_df[pair[0] +'-'+ pair[1]] = [max_loss, R_PNL,profit,optimal_g,optimal_n]
    print("optimisation completed")
    
    for pair in test_pairs:
        tick_data = data_gather_from_files(pair[0],pair[1],currency_pair)['mid']
        max_loss, R_PNL,profit, _ = run_strategy_optimised(tick_data, optimal_g,optimal_n)
        test_df[pair[0] +'-'+ pair[1]] = [max_loss, R_PNL,profit,optimal_g,optimal_n]
    
    train_df = pd.DataFrame(train_df).T
    train_df.columns = ['max_loss', 'R_PNL','profit','optimal_g','optimal_n']
    
    test_df = pd.DataFrame(test_df).T
    test_df.columns = ['max_loss', 'R_PNL','profit','optimal_g','optimal_n']
    return [optimal_g,optimal_n],train_df,test_df

def Profit_Analysis(currency_pair='EURUSD',type ='profit',split='3-1'):
    folder_path = f"results/{currency_pair}/{type}/"
    files = [f for f in os.listdir(folder_path) if f.startswith(split)]
    optimal_file = [f for f in files if f.endswith('optimal.csv')][0]
    test_file = [f for f in files if f.endswith('test.csv')][0]
    train_file = [f for f in files if f.endswith('train.csv')][0]
    optimal_df = pd.read_csv(os.path.join(folder_path, optimal_file),index_col=0)
    test_df = pd.read_csv(os.path.join(folder_path, test_file),index_col=0)
    train_df = pd.read_csv(os.path.join(folder_path, train_file),index_col=0)
    
    for col in optimal_df.columns:
        plt.figure(figsize=(10,5))
        plt.hist(optimal_df[col],bins = min(20,len(optimal_df[col].unique())),align='mid',)
        plt.xticks(rotation=90)
        plt.title(col)
        plt.show()
        
    for col in optimal_df.columns:
        plt.figure(figsize=(20,5))
        plt.plot(optimal_df[col])
        plt.xticks(rotation=90)
        plt.title(col)
        plt.show()

    plt.figure(figsize=(20,5))
    colour_set = ['red','green']
    columns = ['profit','max_loss']
    for col in columns:
        plt.bar(train_df.index,train_df[col],alpha = 0.6,color = colour_set.pop())
    plt.xticks(rotation=90)
    plt.legend(columns)
    plt.show()
    
    plt.figure(figsize=(20,5))
    colour_set = ['red','green']
    columns = ['profit','max_loss']
    for col in columns:
        plt.bar(test_df.index,test_df[col],alpha = 0.6,color = colour_set.pop())
    plt.xticks(rotation=90)
    plt.legend(columns)
    plt.show()
