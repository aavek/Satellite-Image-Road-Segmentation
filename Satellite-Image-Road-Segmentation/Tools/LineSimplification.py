"""
# Reference:
# From https://github.com/mitroadmaps/roadtracer/blob/master/lib/discoverlib/rdp.py
"""

from math import sqrt

def distance(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def point_line_distance(point, start, end):
    if start == end:
        return distance(point, start)
    else:
        n = abs(
            (end[0] - start[0]) * (start[1] - point[1])
            - (start[0] - point[0]) * (end[1] - start[1])
        )
        d = sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        return n / d
        
def Ramer_Douglas_Peucker(points, epsilon):
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d
    if dmax >= epsilon:
        results = Ramer_Douglas_Peucker(points[: index + 1], epsilon)[:-1] + Ramer_Douglas_Peucker(points[index:], epsilon)
    else:
        results = [points[0], points[-1]]
    return results
