import cv2
import numpy as np
import time
import Astar
import Djikstra
import RRT
import Bi_Astar
from bs4 import BeautifulSoup
from PIL import Image

startTime = time.time()
img = cv2.imread('1_identified.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

akaze = cv2.AKAZE_create()
kp_akaze, des_akaze = akaze.detectAndCompute(gray, None)

sift = cv2.SIFT_create(nfeatures=len(kp_akaze)) # The same number of features as in Akaze
kp_sift = sift.detect(gray, None)

orb = cv2.ORB_create(nfeatures=len(kp_akaze))   # The same number of features as in Akaze
kp_orb, des_orb = orb.detectAndCompute(gray, None)

def imageShow(img, keypoints):
    image = cv2.drawKeypoints(img, keypoints, None, color=(0, 200, 0), flags=0)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def pointCoordinates(keypoints):
    return cv2.KeyPoint_convert(keypoints)

### Here goes the creation of the graph edges ###
def graphCreation(pts, startx, starty, finalx, finaly, obstaclesList):
    coordinates = [[]]

    for i in range(len(pts[:, 0])):
        coordinates.append([pts[i, 0], pts[i, 1]])

    coordinates.append([startx, starty])   # Start vertex is n
    coordinates.append([finalx, finaly])   # Final vertex is n+1

    N = len(coordinates)    # Number of vertexes increased after addition of start and final vertexes

    obstacles = []

    minRad = min(np.array(obstaclesList)[:, 2], 0)
    minEdges = []
    for i in range(1, N):
        edgeLen = []
        for j in range(i+1, N+1):
            edgeLen.append(((coordinates[i][0] - coordinates[j][0])**2 +
                            (coordinates[i][1] - coordinates[j][1])**2)**.5)
        minEdges.append(min(edgeLen))

    maxMinDist = max(minEdges)
    medianMinDist = np.median(minEdges)
    meanMinDist = np.mean(minEdges)

    print('MedianMin:', medianMinDist)
    print('MeanMin:', meanMinDist)
    print('Maxmin:', maxMinDist)
    print('MinRad:', minRad)

    edge = max(minRad, meanMinDist)

    for obstacle in obstaclesList:
        for i in range(1, N+1):
            rad = 0.5 * 5**.5 * obstacle[2]
            d = ((obstacle[0] - coordinates[i][0]) ** 2 +
                 (obstacle[1] - coordinates[i][1]) ** 2)**.5
            if d <= rad:
                obstacles.append(i)

    #region GraphConstruction
    graph = []
    for i in range(1, N):
        for j in range(i+1, N+1):
            dist = ((coordinates[i][0] - coordinates[j][0])**2 +
                    (coordinates[i][1] - coordinates[j][1])**2)**.5
            if dist < edge and i not in obstacles and j not in obstacles:
                graph.append([i, j, dist])
    return graph, coordinates, N
    #endregion

def adjacencyListCreation(graph, N):
    # example of adjacency list (or rather map)
    # adjacency_list = 
    # {'1': [('2', 1.2), ('3', 3.4), ('4', 7.9)],
    # '2': [('4', 5.5)],
    # '3': [('4', 12.6)]}
    print('Number of vertexes:', N)
    print('Number of edges:', len(graph), '\n')

    adjacency_list = {i+1: [] for i in range(len(graph))}

    for i in range(len(graph)):
        a, b, ll = map(float, graph[i])
        a, b = int(a), int(b)
        adjacency_list[a].append((b, ll))
        adjacency_list[b].append((a, ll))

    return adjacency_list

def imageSave(img, keypoints, pathList, points, obstaclesList, algoName, cvAlgo):
    image = img.copy()
    # image = cv2.drawKeypoints(image, keypoints, None, color=(0, 200, 0), flags=0)

    f = open('pathList.txt', 'w')
    pathCoords = []
    for v in pathList:
        pathCoords.append(points[v])
        f.write(f'{points[v][0]:.3f} {points[v][1]:.3f}\n')
    pathCoords = np.array(pathCoords, dtype=int)

    for i in range(len(pathCoords) - 1):
        image = cv2.line(image, tuple(pathCoords[i]), tuple(pathCoords[i+1]), (50, 0, 255), 2, lineType = cv2.LINE_AA)

    image = cv2.drawMarker(image, (points[N][0], points[N][1]),
                           (255, 0, 0), 1, markerSize = 12, thickness=3)
    image = cv2.circle(image, (points[N - 1][0], points[N - 1][1]),
                       6, (255, 0, 0), thickness=3)

    obs = image.copy()

    for obstacle in obstaclesList:
        obs = cv2.circle(obs, (int(obstacle[0]),
                                     int(obstacle[1])), obstacle[2], (250, 50, 200), -1)
        newimg = cv2.addWeighted(obs, 0.7, image, 0.3, 0)

    cv2.imwrite('Images/'+algoName+'.png', newimg)
    # print("Execution of "+algoName+" algorithm:", str(round((time.time() - startTime), 2))+'s')

def graphCreation(pts, obstacles_list):
    coordinates = [[]]
    coordinates += pts.copy()

    N = len(coordinates) - 1
    obstacles = []
    minRad = 25
    for obstacle in obstacles_list:
        for i in range(1, N + 1):
            rad = 0.5 * 5 ** .5 * minRad
            d = ((obstacle[0] - coordinates[i][0]) ** 2 +
                 (obstacle[1] - coordinates[i][1]) ** 2) ** .5
            if d <= rad:
                obstacles.append(i)

    # region GraphConstruction
    graph = []
    for i in range(1, N):
        for j in range(i + 1, N + 1):
            dist = ((coordinates[i][0] - coordinates[j][0]) ** 2 +
                    (coordinates[i][1] - coordinates[j][1]) ** 2) ** .5
            if dist < minRad and i not in obstacles and j not in obstacles:
                graph.append([i, j, dist])
    return graph, coordinates, N

def image_discretization(image_path, discretization_step):

    image = Image.open(image_path)
    width, height = image.size

    pts = []
    for i in range(0, width, discretization_step):
        for j in range(0, height, discretization_step):
            pts.append([i, j])
    return pts, width, height


startx = 30
starty = 480
goalx = 256
goaly = 256

sizex, sizey, _ = img.shape

obstaclesList = []
obsctacles = open('Answers.txt', 'r')
lines = obsctacles.readlines()

for line in lines[:1]:
    obstaclesCoord = {}
    for i in range(len(line)):
        if line[i] == '<':
            html = BeautifulSoup(line[i:], features='html.parser')
            html.points.attrs.pop('alt')
            obstaclesCoord = html.points.attrs
            break
    for key in obstaclesCoord.keys():
        if key[0] == 'x':
            x = float(obstaclesCoord[key])
        elif key[0] == 'y':
            y = float(obstaclesCoord[key])
            obstaclesList.append([sizex * x * 0.01, sizey * y * 0.01, 25])

kp_model = kp_akaze
cvAlgo = 'AKAZE'

# print(f'\n--- Feature detection Model: {cvAlgo} ---')
print("\nFINDING THE PATH...")

pts, width, height = image_discretization('1_identified.jpg', discretization_step=5)

pts.append([startx, starty])
pts.append([goalx, goaly])
# pts = pointCoordinates(kp_model) # Needs to be changed for different keypoints

# graph, coordinates, N = graphCreation(pts, startx, starty, goalx, goaly, obstaclesList)
graph, coordinates, N = graphCreation(pts, obstaclesList)
adjacency_list = adjacencyListCreation(graph, N)

pathListAstar = Astar.Graph(adjacency_list).a_star_algorithm(N-1, N, coordinates) # N-1 and N are numbers of start and final points
pathListBiAstar = Bi_Astar.pathplanningBidirectionalAStar(adjacency_list, coordinates, N, N-1, N)
pathListDjikstra = Djikstra.shortestPathFastDjikstra(adjacency_list, N)
pathListRRT = RRT.pathplanningRRT(adjacency_list, coordinates, N, N-1, N)

# Needs to be changed for different keypoints
imageSave(img, kp_model, pathListAstar,  coordinates, obstaclesList, 'A*', cvAlgo)
imageSave(img, kp_model, pathListBiAstar, coordinates, obstaclesList, 'Bi-A*', cvAlgo)
imageSave(img, kp_model, pathListDjikstra, coordinates, obstaclesList, 'Djikstra', cvAlgo)
imageSave(img, kp_model, pathListRRT, coordinates, obstaclesList, 'RRT', cvAlgo)
