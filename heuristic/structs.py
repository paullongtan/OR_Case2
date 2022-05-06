# job structure
import numpy as np
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



