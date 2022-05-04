'''
You do not need to change the code in this file.
You only need to ensure that the TAs can run your algorithm here.
'''
import time
import pandas as pd
import numpy as np
import os
import gurobipy as gp
from gurobipy import GRB
from algorithm_module import heuristic_algorithm

if __name__ == '__main__':
    
    # read all instances (xlsx file) under data folder
    all_data_list = os.listdir('data')
    
    # evaluate all instances 
    result_df = pd.DataFrame(columns = ['Data name', 'Time', 'Number of tardy job', 'Makespan', 'Feasibility'])
    
    for file_name in all_data_list:
        
        start_time = time.time()  
        tardy = np.nan
        makespan = np.nan
        feasibility = False
        try:
            '''
            1. We will import your algorithm here and give you file_path (e.g.,'data/OR110-2_case02_data_s1.csv') as the function argument.
            2. You need to return two lists "machine" and "completion_time".
            '''
            file_path = 'data/' + file_name
            machine, completion_time = heuristic_algorithm(file_path)
            
        except:
            print("the algorithm has errors")

        end_time = time.time()
        spent_time = end_time - start_time
            
        try:
            '''
            We will check feasibility and calculate the objective values here.
            '''
            # feasibility, tardy, makespan = check_feasibility_and_calculate_objective_value(file_path, machine, completion_time)
        except:
            print("the algorithm has errors")
        
        
        result_df = result_df.append({'Data name': file_name, 
                                      'Time': spent_time,
                                      'Number of tardy job': tardy, 
                                      'Makespan': makespan,
                                      'Feasibility': feasibility}, ignore_index = True)

# output result
result_df.to_csv('result.csv', index = False)