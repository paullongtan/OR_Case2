#!/usr/bin/env python
# coding: utf-8
def heuristic_algorithm(fullpath):
    import os
    import pandas as pd
    import numpy as np
    from heapq import heappush, heappop, heapify
    # job structure
    GAP = 1e-7
    class Job:
        '''structure for 1 job '''
        def __init__(self, row):
            '''input := df.iloc[idx, :]'''
            self.id = row['Job ID']
            self.due = row['Due Time']
            self.next_op = 0 # True as complete, False as not yet processed
            self.stage_pt = [row['Stage-1 Processing Time'], row['Stage-2 Processing Time']]
            mfor1 = list(map(int, row['Stage-1 Machines'].split(',')))
            if row['Stage-2 Machines'] is not np.nan:
                mfor2 = list(map(int, row['Stage-2 Machines'].split(',')))
            else: mfor2 = []
            self.stage_mach = [mfor1, mfor2]
            self.assign_mach = [None for _ in range(2)]
            self.start_time = [-1 for _ in range(2)]
            self.end_time = [-1 for _ in range(2)]

        def __repr__(self):
            return f'\
              * Job id: {self.id}\n\
              * Due time:{self.due}\n\
              stage 1: {self.assign_mach[0]}\n\
                       {self.stage_pt[0]}, {self.stage_mach[0]}\n\
              stage 2: {self.assign_mach[1]}\n\
                       {self.stage_pt[1]}, {self.stage_mach[1]}'
        __str__ = __repr__

    class Jobs:
        '''structure for multiple jobs' management'''
        def __init__(self, n):
            self.completion_times = np.zeros(n)
            self.tardiness = np.zeros(n)
            self.is_completed = np.full(n, False)
            self.residual_times = np.zeros(n)
            self.jobs = []

        def get_RRDD(self):
            if getattr(self, 'RRDD', None) is None:
                self.RRDD = self.due_dates - np.min(self.due_dates)
            return self.RRDD # static

        def get_LST(self):
            '''latest start time'''
            self.LST = self.due_dates - self.residual_times
            return self.LST

        def add_jobs(self, data):
            self.due_dates = data['Due Time'].to_numpy()
            for i in range(len(data)):
                row = data.iloc[i, :]
                jobi = Job(row)
                self.residual_times[i] = sum(jobi.stage_pt)
                self.jobs.append(jobi)


        def assign(self, job_name, mach, st):
            '''job_name = (2, 0) means job 3 and op 1
            note that job and op is 0-indexed as well as machines
            op
            '''
            jobidx, op = job_name
            job = self.jobs[jobidx]
            self.completion_times[jobidx] = st + job.stage_pt[op]
            self.residual_times[jobidx] -= job.stage_pt[op]
            job.assign_mach[op] = mach
            job.start_time[op] = st
            job.end_time[op] = self.completion_times[jobidx]
            job.next_op = op+1
            if op == 1:
                self.is_completed[jobidx] = True

    class Machines:
        def __init__(self, df):
            '''pass the stage1, stage2 machine lists'''
            mfor1 = df['Stage-1 Machines'].values.tolist()
            mfor2 = df['Stage-2 Machines'].values.tolist()
            mfor1 = [list(map(int, x.split(','))) for x in mfor1]
            # print('mfor2', mfor2, type(mfor2))
            mfor2 = [list(map(int, x.split(','))) for x in mfor2 if x is not np.nan]
            mfor1 = [item for sublist in mfor1 for item in sublist]
            mfor2 = [item for sublist in mfor2 for item in sublist]

            self.number = max(max(mfor1), max(mfor2))
            self.versatile = [mfor1.count(i+1) + mfor2.count(i+1) for i in range(self.number)]
            self.holes = [[] for _ in range(self.number)]

            self.fintime = [0 for _ in range(self.number)]
            self.HL = [0 for _ in range(self.number)]


        def _schedule(self, mach, job_name, st, proc_time):
            '''mach is 0-indexed'''
            display_name = tuple([x+1 for x in job_name])
            # self.schedule[mach].append((f'{display_name}', round(end_time, 3)))
            self.fintime[mach] = st + proc_time

        def add_idle(self, mach, idle_time,
                    hole_start, hole_end):
            # self.schedule[mach].append((f'idle', round(hole_end)))
            self.fintime[mach] += idle_time
            # add hole
            # len(self.schedule)-1 is the idx of this hole in schedule
            hole = tuple([hole_start, hole_end])
            self.holes[mach].append(hole)
            self.HL[mach] += hole_end - hole_start

        def schedule_hole(self, job_name, mach, hole_id, fill_end):
            '''update and return updated average hole length'''
            hole = self.holes[mach][hole_id]
            hole_start, hole_end = hole
            hole_length = hole_end - hole_start
            if abs(hole_end - fill_end) < GAP:
                # pop the hole by hole_id
                self.holes[mach].pop(hole_id)
            else:
                self.holes[mach][hole_id] = tuple([fill_end, hole_end])
            self.HL -= (fill_end - hole_start) # processing time
            # display_name = tuple([x+1 for x in job_name])
            return self.HL/len(self.holes)

    def make_Q(J):
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


    def find_hole(job_name,
              currjob,
              J,
              M,
              Mach_Q,
              AvailMachTable):
        curr_jindex, curr_op = job_name
        proc_time = currjob.stage_pt[curr_op]
        def find_hole_helper():
            for idx, curritem in enumerate(Mach_Q):
                _, m_id = curritem
                if not AvailMachTable[curr_jindex][curr_op][m_id]:
                    continue
                # find legal holes
                if not M.holes[m_id]:
                    continue
                for hole_id, hole in enumerate(M.holes[m_id]):
                    hole_start, hole_end = hole
                    hole_length = hole_end - hole_start
                    # enough length
                    if hole_length - proc_time >= -GAP:
                        # check legal precedence
                        curr_hole_id = hole_id
                        if curr_op == 1 and hole_start >= currjob.end_time[0]:
                            return idx, m_id, curr_hole_id, hole_start + proc_time
                        elif curr_op == 0:
                            return idx, m_id, curr_hole_id, hole_start + proc_time
                    # 一律從hole_start開始schedule，沒辦法的話就跳過（不然更新holes那邊變超麻煩）
        res = find_hole_helper()
        if not res:
            # print(f'No result in finding a hole for {[x+1 for x in job_name]}')
            return False
        if res:
            idx, m_id, hole_id, fill_end = res
            #print(f'Schduling {m_id+1}, {M.holes[m_id][hole_id]} for {[x+1 for x in job_name]}')
            #print(f'Original: {M.holes[m_id]}')
            # idx是Queue中machine的位置
            hole = M.holes[m_id][hole_id]
            # print(hole)
            hole_start, hole_end = hole
            J.assign(job_name = job_name,
                    mach = m_id,
                    st = hole_start)
            # update hole length and replace avg_hole_length
            new_avg_hl = M.schedule_hole(
                    job_name = job_name,
                    mach = m_id,
                    hole_id = hole_id,
                    fill_end = fill_end)
            # print(f'Updated: {M.holes[m_id]}')
            return True

    def heuristic(J, M):
        # best_makepsan = sum(job_processing_time) for all jobs / |M|
        # heperparameters
        TOLRATIO = 0.3
        Fail_Tolerance = 2
        best_makespan = sum(job.stage_pt[0]+job.stage_pt[1] for job in J.jobs)/M.number
        tolerance = best_makespan * TOLRATIO  # tolerance for idle time, if idle > tolerance, do not schedule the curr op in the current epoch.
        #print(f'[INFO] {len(J.jobs)} jobs, {M.number} machines')
        #print(f'[INFO] Tolerance: {tolerance:.2f}')

        # Job_Q (lst_ratio, job_index, job_op)
        Job_Q = make_Q(J)
        # print(f'Job Queue: {Job_Q}')
        # Mach_Q (versatility, avg_hole_length, m)
        Mach_Q = make_mQ(M)
        AvailMachs = getAvailMachs(J = J, M = M)


        fails = [0 for _ in range(len(J.jobs))]
        epoch = 0
        while Job_Q:
            epoch += 1
            PERMIT = True
            # step 3. extract_min() to get the job with minimal LST and its other attributes
            _, curr_job_index, curr_op = heappop(Job_Q)
            curr_job = J.jobs[curr_job_index]
            op_proc_time = curr_job.stage_pt[curr_op]
            job_name = (curr_job_index, curr_op)


            # if curr_job has no second operation
            if op_proc_time <= GAP and curr_op == 1:
                J.assign(job_name = job_name,
                        mach = None,
                        st = curr_job.end_time[curr_op-1])
                # note that it's only possible for second operation to have proc time = 0
                # so this doesn't trigger index error
                continue
            # step 4-1. calculate the best machine: find-hole
            # 'job_name', 'currjob', and 'Mach_Q'
            if find_hole(J = J, M = M,
                         job_name = job_name, currjob = curr_job, Mach_Q = Mach_Q,
                      AvailMachTable = AvailMachs):

                continue


            # step 4-2. if find-hole fails, calculate the best machine and schedule at the end
            avail_machines_idx = [x-1 for x in curr_job.stage_mach[curr_op]]
            curr_machine = min(avail_machines_idx, key = lambda x: (M.fintime[x], M.versatile[x], x))
            # ARE THERE REASONS TO POSTPONE THE CURR OP?
            if J.completion_times[curr_job_index] + op_proc_time > J.due_dates[curr_job_index] and curr_op == 1 and fails[curr_job_index] < Fail_Tolerance:
                #print(f'[INFO] Job {curr_job_index+1} op {curr_op+1} will be tardy even if scheduled, queue last.')
                curr_new_value = float('inf')
                PERMIT = False

            # ARE THERE REASONS TO POSTPONE THE CURR OP (if curr_op is second op)?
            elif M.fintime[curr_machine] < J.completion_times[curr_job_index]:

                idle = J.completion_times[curr_job_index] - M.fintime[curr_machine]
                if idle > tolerance and curr_op == 1 and fails[curr_job_index] < Fail_Tolerance:
                    #print(f'[INFO] Job {curr_job_index+1} op {curr_op+1} has idle {idle:.2f}, postpone it.')
                    PERMIT = False
                    if Job_Q:
                        curr_new_value = Job_Q[0][0] + 3
                    else:
                        curr_new_value = 0 # the last one
                else:
                    #print(f'[INFO] Job {curr_job_index+1} op {curr_op+1} has idle {idle:.2f}.')

                    M.add_idle(
                    hole_start = M.fintime[curr_machine],
                    hole_end =  J.completion_times[curr_job_index],
                    mach = curr_machine,
                    idle_time = idle)

            if PERMIT:
                # print(f'Scheduling {[x+1 for x in job_name]} on machine {curr_machine+1}\'s end at {M.fintime[curr_machine]}')
                J.assign(job_name = job_name,
                    mach = curr_machine,
                     st = M.fintime[curr_machine]
                    )
                M._schedule(job_name = job_name,
                   mach = curr_machine,
                    proc_time = op_proc_time,
                   st = M.fintime[curr_machine])
                curr_new_value = J.get_LST()[curr_job_index]
            else:
                fails[curr_job_index] += 1
            # print(f'{epoch} Fails Count:', fails)
            # update the LST value and push it back to Q if the job has its second operation that hasn't been done
            if not PERMIT:
                heappush(Job_Q, (curr_new_value, curr_job_index, curr_op))
                # it maintains the heap invariant, no need to heapify
        return J, M

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

    instances = []
    instances.append(pd.read_csv(fullpath))

    df = instances[0]
    M = Machines(df)
    J = Jobs(len(df))
    J.add_jobs(df)
    heuristic(J = J, M = M)
    Tardy_jobs = list(np.where(J.completion_times > J.due_dates)[0])
    Tardy_jobs = [x+1 for x in Tardy_jobs]
    Makespan = max(M.fintime)
    Answers = list(make_result(J))
    return Answers