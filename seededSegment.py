import numpy as np
import cv2
from tqdm import tqdm
from queue import Queue


def checkPoint(img, point, visited):
    if 0 <= point[0] < img.shape[0] and 0 <= point[1] < img.shape[1]:
        if visited[point[0], point[1]] == 1:
            return 0
        return 1
    return 0


def updatePixel(img, point, tolerance, result, average, count):
    B = abs(int(img[point[0], point[1], 0]) - average[0] / count)
    G = abs(int(img[point[0], point[1], 1]) - average[1] / count)
    R = abs(int(img[point[0], point[1], 2]) - average[2] / count)

    if B < tolerance and G < tolerance and R < tolerance:
        result[point[0], point[1], 0] = img[point[0], point[1], 0]
        result[point[0], point[1], 1] = img[point[0], point[1], 1]
        result[point[0], point[1], 2] = img[point[0], point[1], 2]
        average[0] += float(img[point[0], point[1], 0])
        average[1] += float(img[point[0], point[1], 1])
        average[2] += float(img[point[0], point[1], 2])
        count += 1
        return 1, result, average, count
    return 0, result, average, count


def ShowResults(filename, result):
    cv2.imwrite(filename, result)


def SeedPointSegmentation(img, seedPoint, tolerance, imgNameOut):
    Tolerance = tolerance
    Visited = np.zeros((img.shape[0], img.shape[1]), dtype=int)
    Result = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    Average = [float(img[seedPoint[0], seedPoint[1], 0]), float(img[seedPoint[0], seedPoint[1], 1]), float(img[seedPoint[0], seedPoint[1], 2])]
    Count = 1

    Q = Queue()
    Q.put([seedPoint[0], seedPoint[1]])
    Visited[seedPoint[0], seedPoint[1]] = 1

    while Q.qsize() > 0:
        CurrentPoint = Q.get()
        Ret, Result, Average, Count = updatePixel(img, CurrentPoint, Tolerance, Result, Average, Count)
        if Ret == 0:
            continue

        p1 = [CurrentPoint[0] + 1, CurrentPoint[1]]
        p2 = [CurrentPoint[0] - 1, CurrentPoint[1]]
        p3 = [CurrentPoint[0], CurrentPoint[1] - 1]
        p4 = [CurrentPoint[0], CurrentPoint[1] + 1]

        if checkPoint(img, p1, Visited):
            Q.put(p1)
            Visited[p1[0], p1[1]] = 1
        if checkPoint(img, p2, Visited):
            Q.put(p2)
            Visited[p2[0], p2[1]] = 1
        if checkPoint(img, p3, Visited):
            Q.put(p3)
            Visited[p3[0], p3[1]] = 1
        if checkPoint(img, p4, Visited):
            Q.put(p4)
            Visited[p4[0], p4[1]] = 1

    ShowResults(imgNameOut, Result)

ImageList = ["face1.jpg", "face2.jpg", "face3.jpg", "face4.jpg"]

test_configs = [
    {"image": ImageList[0], "seedPoint": [600, 950], "tolerance": 100},
    {"image": ImageList[1], "seedPoint": [600, 950], "tolerance": 100},
    {"image": ImageList[2], "seedPoint": [600, 950], "tolerance": 100},
    {"image": ImageList[3], "seedPoint": [600, 950], "tolerance": 100},
]

for i, config in enumerate(test_configs):
    Image = cv2.imread(config["image"])
    if Image is not None:
        print(f"Processing {config['image']} with seed point {config['seedPoint']} and tolerance {config['tolerance']}")
        output_filename = f"out_seed_seg_{i+1}.png"
        SeedPointSegmentation(Image, config["seedPoint"], config["tolerance"], output_filename)
    else:
        print(f"Failed to load image {config['image']}")

