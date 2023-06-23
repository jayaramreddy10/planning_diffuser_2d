import numpy as np

with open('trajectories.txt', 'r') as file:
    data = file.read()

trajs = []
samples_start = data.find('samples = [[')

while samples_start != -1:
    samples_end = data.find(']]', samples_start)
    samples_str = data[samples_start + 11:samples_end]

    samples_lines = samples_str.split('\n')
    samples = [list(line.strip().split("\t")) for line in samples_lines if line.strip()]

    for i in range(len(samples)):
        s = samples[i][0]
        s = s[1:len(s)-1]
        s = s.split()
        s = (float(s[0]) , float(s[1]))
        samples[i] = s

    trajs.append((np.array(samples), {0 : np.array(samples[0]), len(samples) - 1: np.array(samples[-1])}))
# cond_list = []
# cond_start = data.find('cond = {')
    samples_start = data.find('samples = [[', samples_end)

print(trajs)