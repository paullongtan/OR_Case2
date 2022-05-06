def get_machine_number(df):
    import numpy as np
    '''helper function of checker()'''
    mfor1 = df['Stage-1 Machines'].values.tolist()
    mfor2 = df['Stage-2 Machines'].values.tolist()
    mfor1 = [list(map(int, x.split(','))) for x in mfor1]
    mfor2 = [list(map(int, x.split(','))) for x in mfor2 if x is not np.nan]
    mfor1 = [item for sublist in mfor1 for item in sublist]
    mfor2 = [item for sublist in mfor2 for item in sublist]
    return max(max(mfor1), max(mfor2))


def checker(ans, instance):
    '''
    Given ans as (machine, completion_time)
        machine := a list of list of machines; machine[i][j] is job i+1 op j+1 's assigned machine
        completion_time := a list of list of completion times; completion_time[i][j] is job i+1 op j+1 's compleiton time
    Check the schedule's feasibility:
    (1) precedence of same job's operations
    (2) assigned-machine's availability (if it is capable of doing the job's op)
    (3) whether operations on each machine have overlap
    (4) if (1) ~ (3) are passed, compute the 2 objectives and return case passed; otherwise failed.
    '''
    import numpy as np
    gap = 1e-4
    n = len(instance)
    GAP = np.zeros((n))
    GAP.fill(gap)

    M, C_times = ans
    M, C_times = np.array(M), np.array(C_times)
    M_number = get_machine_number(instance)
    # 1. check job precedence
    op1_ends = C_times[:, 0]
    op2_ends = C_times[:, 1]
    op1_pt = instance['Stage-1 Processing Time'].to_numpy()
    op2_pt = instance['Stage-2 Processing Time'].to_numpy()
    op_pts = np.column_stack((op1_pt, op2_pt))
    op_ends = np.column_stack((op1_ends, op2_ends))


    mfor1 = instance['Stage-1 Machines'].to_numpy()
    mfor2 = instance['Stage-2 Machines'].to_numpy()
    AvailMachs = np.column_stack((mfor1, mfor2))

    # print(op_ends)
    x = op1_ends + op2_pt
    # print('[Zero-index!]')
    try:
        assert np.all((op1_ends + op2_pt - op2_ends) <= GAP)
    except AssertionError:
        print('Operation precedence within the same job violated')
        print('op1 ends + op2 proc time', op1_ends + op2_pt)
        print('op2 ends (job completion times)', op2_ends)
        return
    # 2. re-construct each machine schedule
    sch = [[] for _ in range(M_number)]
    for i in range(n):
        for j in range(2):
            # this machine is 1-indexed
            machine = M[i][j]
            # 3. machine check: availability
            if machine is None:
                continue
            if str(machine) not in AvailMachs[i, j].split(','):
                print(f'Job {(i+1, j+1)} is not scheduled on an allowed machine.')
                return

            op_end = op_ends[i][j]
            # sch 
            sch[machine-1].append(((i+1,j+1), op_end))

    sch = [sorted(sc, key = lambda x:x[1]) for sc in sch]
    # print(*sch, sep = '\n')
    # print(op_pts)
    # 4. check if any operations overlap
    for mi, sc in enumerate(sch):
        i = len(sc)-1
        while i > 0:
            (currop, currop_ct), (prevop, prevop_ct) = sc[i], sc[i-1]
            currop_pt = op_pts[currop[0]-1, currop[1]-1]
            if currop_ct - currop_pt - prevop_ct < -gap:
                print(f'Job {[x+1 for x in currop]} and {[x+1 for x in prevop]} overlap on machine {mi+1}')
                print('prevop completion time', prevop_ct)
                print('currop processing time', currop_pt)
                print('currop completion time', currop_ct)
                return
            i -= 1
    # 5. OBJ: compute tardy
    dues = instance['Due Time'].to_numpy()
    tardy = list(np.where(op2_ends > dues)[0])
    # 6. OBJ: compute makespan
    # print(sch)
    m_ends = [m_sche[-1][-1] for m_sche in sch if len(m_sche) > 0]
    makespan = max(m_ends)
    print('REVISED!!!!')
    tardy = [x+1 for x in tardy]
    
    return tardy, makespan, sch