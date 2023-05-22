import math
import numpy as np

def lerp(v0, v1, i):
    return v0 + i * (v1 - v0)

def getEquidistantPoints(p1, p2, n):
    if(n == 0):
        return []
    return [(lerp(p1[0],p2[0],1./n*i), lerp(p1[1],p2[1],1./n*i)) for i in range(n+1)]

# 384 points from samples

def generate_samples(path,no_samples=384,pathres=0.02):
    pathlen = len(path)
    res_samples = []
    for i in range(pathlen - 1):
        (x1,y1) = path[i]
        (x2,y2) = path[i+1]

        dist = round(math.sqrt( (x1-x2)**2 + (y1-y2)**2  ),2)

        if(dist == 0):
            continue

        np = int(round(dist/pathres,0))

        res_samples = res_samples + getEquidistantPoints(path[i], path[i+1], np)

    p = len(res_samples)


    if(no_samples>=p):

        n = int((no_samples-p)/(p-1))

        k = (no_samples-p)%(p-1)

        samples = []
        for i in range(p - 1):
            if(i<=k):
                samples = samples + getEquidistantPoints(res_samples[i], res_samples[i+1], n + 1 )
            else:
                samples = samples + getEquidistantPoints(res_samples[i], res_samples[i+1], n )

        return (samples,p)
    
    else:

        delete = p - no_samples

        del_inds = round(np.linspace(0,p,delete),0)

        # print("deleting ",del_inds,"indexes","from ",p)

        [res_samples.pop(inds) for inds in del_inds]

        return (res_samples,p)
    

