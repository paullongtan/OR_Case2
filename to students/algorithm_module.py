def heuristic_algorithm(file_path):
    
    '''
    1. Write your heuristic algorithm here.
    2. We would call this function in CA2_grading_program.py to evaluate your algorithm.
    3. Please do not change the function name and the file name.
    4. The parameter is the file path of a data file, whose format is specified in the document. 
    5. You need to return your schedule in two lists "machine" and "completion_time".
        (a) machine[j][0] is the machine ID of the machine to process the first stage of job j + 1, and 
            machine[j][1] is the machine to process the second stage of job j + 1.
        (b) completion_time[j][0] is the completion time of the first stage of job j + 1, and 
            completion_time[j][1] is the completion time of the second stage of job j + 1. 
        Note 1. If you have n jobs, both the two lists are n by 2 (n rows, 2 columns). 
        Note 2. In the list "machine", you should record the IDs of machines 
                (i.e., to let machine 1 process the first stage of job 1, 
                you should have machine[0][0] == 1 rather than machine[0][0] == 0).
    6. You only need to submit this algorithm_module.py.
    '''
    
    import csv
    
    # read data and store the information into your self-defined variables
    fp = open(file_path, 'r', newline = '')
    header = fp.readline() 
    reader = csv.reader(fp, delimiter = ',')
    # for a_row in reader:
    #     print(a_row) # a_row is a list
    # ...
    
    
    

    # start your algorithm here
    machine = []
    completion_time = []
    # ...  
    
    
    
    return machine, completion_time