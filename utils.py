GAP = 1e-7
def make_Q(J):
    from heapq import heappush, heappop, heapify
    Job_keys = []
    for job_index in range(len(J.jobs)):
        op1_pt, op2_pt = J.jobs[job_index].stage_pt
        # 我覺得 due_dates 幾乎等於 total proc time的反而可以放後面(不太可能來得及，乾脆果斷放棄)
        LST = J.get_LST()
        if LST[job_index] <= GAP: # avoid division by zero error
            lst_ratios = [0, 0]
        else:
            lst_ratios = [op1_pt / J.LST[job_index], op2_pt / J.LST[job_index]]
        # preserve precedence:
        lst_ratios[0] = max(lst_ratios)
        # NOTE: min queue extracts by minimum value, so add a negative sign here
        # (lst_ratio, job_index, job_op)
        Job_keys.append((-lst_ratios[0], job_index, 0))
        Job_keys.append((-lst_ratios[1], job_index, 1))
    Q = Job_keys[:]
    heapify(Q)
    return Q

def getAvailMachs(J, M):
    import numpy as np
    bigM = M.number
    bigN = len(J.jobs)
    AvailMachs = np.full((bigN, 2, bigM),
                         fill_value = False)
    for i in range(bigN):
        mfor1, mfor2 = J.jobs[i].stage_mach[0], J.jobs[i].stage_mach[1]
        for m in mfor1:
            AvailMachs[i][0][m-1] = True
        for m in mfor2:
            AvailMachs[i][1][m-1] = True
    # indexing AvailMachs[i][j][m] to check if job (i,j) can be put on machine m (all ZERO-indexed)
    return AvailMachs

def make_mQ(M):
    Q = []
    for m in range(M.number):
        versatility = M.versatile[m]
        Q.append([versatility, m])
    return sorted(Q, key = lambda x:(x[0], x[1]))


def make_result(J):
    '''pass in the resulted J'''
    n = len(J.jobs)
    machine, completion_time = [], []
    for i in range(n):
        if  J.jobs[i].assign_mach[1] is None:
            op2_mach_id = None
        else: op2_mach_id = J.jobs[i].assign_mach[1]+1
        op1_mach_id = J.jobs[i].assign_mach[0]+1
        op1_c_time = round(J.jobs[i].end_time[0], 3)
        op2_c_time = round(J.jobs[i].end_time[1], 3)
        machine.append([op1_mach_id, op2_mach_id])
        completion_time.append([op1_c_time, op2_c_time])
    assert len(machine) == len(completion_time) == n
    return machine, completion_time