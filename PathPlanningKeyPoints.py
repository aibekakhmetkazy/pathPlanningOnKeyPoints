import cv2
import numpy as np
import time
import Astar
import Djikstra
# import Dstar_lite

startTime = time.time()
img = cv2.imread('high_resolution_image16.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

akaze = cv2.AKAZE_create()
kp_akaze, des_akaze = akaze.detectAndCompute(gray, None)
kp_image_akaze = cv2.drawKeypoints(img, kp_akaze, None, color=(0, 200, 0), flags=0)

sift = cv2.SIFT_create(nfeatures=len(kp_akaze)) # The same number of features as in Akaze
kp_sift = sift.detect(gray, None)
kp_image_sift = cv2.drawKeypoints(img, kp_sift, None, color=(0, 200, 0), flags=0)

orb = cv2.ORB_create(nfeatures=len(kp_akaze))   # The same number of features as in Akaze
kp_orb, des_orb = orb.detectAndCompute(gray, None)
print(len(kp_orb))
kp_image_orb = cv2.drawKeypoints(img, kp_orb, None, color=(0, 200, 0), flags=0)

def imageShow(image):
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def pointCoordinates(keypoints):
    return cv2.KeyPoint_convert(keypoints)

### Here goes the creation of the graph edges ###
def graphCreation(pts, startx, starty, finalx, finaly, obstaclesList):
    coordinates = [[]]
    n = len(pts[:, 0])

    for i in range(n):
        coordinates.append([pts[i, 0], pts[i, 1]])

    coordinates.append([startx, starty])   # Start vertex is n
    coordinates.append([finalx, finaly])   # Final vertex is n+1

    N = n+2     # Number of vertexes increased after addition of start and final vertexes

    obstacles = []

    minRad = min(np.array(obstaclesList)[:, 2])

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
            if dist < minRad and i not in obstacles and j not in obstacles:
                graph.append([i, j, dist])
    return graph, coordinates, N
    #endregion

def adjacencyListCreation(graph, N):
    # example of adjacency list (or rather map)
    # adjacency_list = 
    # {'1': [('2', 1.2), ('3', 3.4), ('4', 7.9)],
    # '2': [('4', 5.5)],
    # '3': [('4', 12.6)]}
    print('Number of keypoints:', N)
    print('Number of edges:', len(graph), '\n')

    adjacency_list = {i+1: [] for i in range(len(graph))}

    for i in range(len(graph)):
        a, b, ll = map(float, graph[i])
        a, b = int(a), int(b)
        adjacency_list[a].append((b, ll))
        adjacency_list[b].append((a, ll))

    return adjacency_list

def imageSave(image, pathList, points, obstaclesList, algoName, cvAlgo, zoom):

    f = open('pathList.txt', 'w')
    pathCoords = []
    for v in pathList:
        pathCoords.append(points[v])
    f.write(str(np.array(pathCoords, dtype=float)) + '\n')
    pathCoords = np.array(pathCoords, dtype=int)

    for i in range(len(pathCoords) - 1):
        image = cv2.line(image, tuple(pathCoords[i]), tuple(pathCoords[i+1]), (50, 0, 255), 2, lineType = cv2.LINE_AA)

    image = cv2.drawMarker(image, (points[N][0], points[N][1]),
                           (220, 220, 220), 1, markerSize = 12, thickness=3)
    image = cv2.circle(image, (points[N - 1][0], points[N - 1][1]),
                       6, (200, 200, 200), thickness=3)

    obs = image.copy()

    for obstacle in obstaclesList:
        obs = cv2.circle(obs, (int(obstacle[0]),
                                     int(obstacle[1])), obstacle[2], (250, 50, 200), -1)
        newimg = cv2.addWeighted(obs, 0.7, image, 0.3, 0)

    cv2.imwrite('Images/'+algoName+cvAlgo+zoom+'.png', newimg)
    print("Execution of "+algoName+" algorithm:", str(round((time.time() - startTime), 2))+'s')

startx = 200
starty = 200
goalx = 370
goaly = 840

obstaclesList = [[300, 600, 60], [450, 700, 60], [300, 750, 60]]  # List of given circle obstacles with x,y,r values

pts = pointCoordinates(kp_orb)
graph, coordinates, N = graphCreation(pts, startx, starty, goalx, goaly, obstaclesList)

adjacency_list = adjacencyListCreation(graph, N) # pts and x,y of start and final points

pathListDjikstra = Djikstra.shortestPathFastDjikstra(adjacency_list, N)
pathListAstar = Astar.Graph(adjacency_list).a_star_algorithm(N-1, N, coordinates) # N-1 and N are numbers of start and final points

imageSave(kp_image_orb, pathListDjikstra, coordinates, obstaclesList, 'Djikstra', cvAlgo='Orb', zoom='16')
imageSave(kp_image_orb, pathListAstar,  coordinates, obstaclesList, 'Astar', cvAlgo='Orb', zoom='16')