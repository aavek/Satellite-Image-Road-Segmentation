"""
# Reference:
# https://github.com/anilbatra2185/road_connectivity
"""

import math
import numpy as np

def segment_to_linestring(segment):
    if len(segment) < 2:
        return []
    linestring = "LINESTRING ({})"
    sublinestring = ""
    for i, node in enumerate(segment):
        if i == 0:
            sublinestring = sublinestring + "{:.1f} {:.1f}".format(node[1], node[0])
        else:
            if node[0] == segment[i - 1][0] and node[1] == segment[i - 1][1]:
                if len(segment) == 2:
                    return []
                continue
            if i > 1 and node[0] == segment[i - 2][0] and node[1] == segment[i - 2][1]:
                continue
            sublinestring = sublinestring + ", {:.1f} {:.1f}".format(node[1], node[0])
    linestring = linestring.format(sublinestring)
    return linestring


def segments_to_linestrings(segments):
    linestrings = []
    for segment in segments:
        linestring = segment_to_linestring(segment)
        if len(linestring) > 0:
            linestrings.append(linestring)
    if len(linestrings) == 0:
        linestrings = ["LINESTRING EMPTY"]
    return linestrings


def uniqueLinestrings(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list
    
def Graph_to_Keypoints(graph_label_road_segments):
    linestrings = uniqueLinestrings(segments_to_linestrings(graph_label_road_segments))
    keypoints = []
    for line in linestrings:
        linestring = line.rstrip("\n").split("LINESTRING ")[-1]
        points_str = linestring.lstrip("(").rstrip(")").split(", ")
        ## If there is no road present
        if "EMPTY" in points_str:
            return keypoints
        points = []
        for pt_st in points_str:
            x, y = pt_st.split(" ")
            x, y = float(x), float(y)
            points.append([x, y])

            x1, y1 = points[0]
            x2, y2 = points[-1]
            zero_dist1 = math.sqrt((x1) ** 2 + (y1) ** 2)
            zero_dist2 = math.sqrt((x2) ** 2 + (y2) ** 2)

            if zero_dist2 > zero_dist1:
                keypoints.append(points[::-1])
            else:
                keypoints.append(points)
    return keypoints