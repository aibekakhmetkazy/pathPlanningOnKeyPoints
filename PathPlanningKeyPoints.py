import cv2
import heapq
import numpy as np
import matplotlib.pyplot as plt
import time

startTime = time.time()
img = cv2.imread('high_resolution_image17.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initiate SIFT object with default values
sift = cv2.SIFT_create(nfeatures=4103)
# keypoints = sift.detect(gray, None)
# kp_image = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)
# cv2.imwrite('sift.jpg', img1)

# # Applying the ORB function 
orb = cv2.ORB_create(nfeatures=4103) 
# keypoints, des = orb.detectAndCompute(gray, None) 
# kp_image = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0) 
# cv2.imwrite('orb.jpg', kp_image)

akaze = cv2.AKAZE_create()
keypoints, descriptors = akaze.detectAndCompute(gray, None)
kp_image = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)
# cv2.imshow('Akaze', kp_image)
# cv2.waitKey(0)
# cv2.imwrite('akaze.jpg', kp_image)

pts = cv2.KeyPoint_convert(keypoints)
x, y = pts[:, 0], pts[:, 1]

### Here goes the creation of the graph edges ###

def graphCreation(pts, startx, starty, finalx, finaly):
    pointsAndCoordinates = [[]]
    n = len(pts[:, 0])

    for i in range(1, n+1):
        pointsAndCoordinates.append([pts[i-1, 0], pts[i-1, 1]])

    pointsAndCoordinates.append([startx, starty])   # Start vertex is n
    pointsAndCoordinates.append([finalx, finaly])   # Final vertex is n+1

    N = n+2     # Number of vertexes increased after addition of start and final vertexes

    obstaclesList = [[300, 600, 50], [450, 700, 50], [300, 750, 50]]  # List of given circle obstacles with x,y,r values
    obstacles = []

    minRad = 50

    for obstacle in obstaclesList:
        if obstacle[2] < minRad:
            minRad = obstacle[2]
        for i in range(1, N+1):
            rad = np.sqrt(5) / 2 * obstacle[2]
            d = round(((obstacle[0] - pointsAndCoordinates[i][0]) ** 2 +
                       (obstacle[1] - pointsAndCoordinates[i][1]) ** 2) ** .5, 2)
            if d <= rad:
                obstacles.append(i)

    #region File
    # For execution with different conditions and to form list of edges this region has to be uncommented ###
    adjList = []
    fileStart = time.time()
    with open('points16.txt', 'w') as f:  # Writing all the vertexes and distances into 1 txt file for matrix construction
        for i in range(1, N):
            for j in range(i+1, N+1):
                dist = round(((pointsAndCoordinates[i][0] - pointsAndCoordinates[j][0])**2 +
                        (pointsAndCoordinates[i][1] - pointsAndCoordinates[j][1])**2)**.5, 2)
                if dist < minRad and i not in obstacles and j not in obstacles:
                    adjList.append([i, j, dist])
                    # f.write("{} {} {}\n".format(i, j, dist))
        f.write("\n".join([" ".join(map(str, a)) for a in adjList]))
    print('.txt file creation took:',round(time.time()-fileStart, 2))
    #endregion
    return pointsAndCoordinates, obstaclesList, N

pointsAndCoordinates, obstaclesList, N = graphCreation(pts, 100, 100, 500, 900)

### Here goes the Djikstra algo for path planning ###
#region Djikstra
def shortestPathFastDjikstra(N):
    f = open('points16.txt')
    lines = f.read().splitlines()

    print('\nNumber of edges:', len(lines))
    print('Number of vertexes: ', N)

    prev = [-1]*(N+1)
    m = [[] for _ in range(N + 1)]    # For the adjacency list, create an empty list
    for i in range(len(lines)):       # Each array of adjacency list is filled according to the list of edges
        a, b, ll = map(float, lines[i].split())
        a, b = int(a), int(b)
        m[a].append([b, ll])
        m[b].append([a, ll])

    s, f = N-1, N
    pathByVertexes = [f]

    visited = [False] * (N + 1)
    dist = [[float('inf'), i] for i in range(N + 1)]
    dist[s][0] = 0
    heap = [[0, s]]

    while len(heap) > 0 and not visited[f]:
        visited[s] = True
        s = heap[0][1]
        for j in range(len(m[s])):
            if dist[m[s][j][0]][0] > dist[s][0] + m[s][j][1]:
                dist[m[s][j][0]][0] = dist[s][0] + m[s][j][1]
                prev[m[s][j][0]] = s
                if not visited[m[s][j][0]]:
                    heapq.heappush(heap, dist[m[s][j][0]])
        s = heapq.heappop(heap)[1]

    if dist[f][0] == float('inf'):
        print(-1)
    else:
        print('\nDistance to cover:', round(dist[f][0], 3))

    if dist[f][0] == float('inf'):
        print('There is no path to the final vertex!')
    else:
        def path(prev, pathByVertexes, f):
            if prev[f] > -1:
                pathByVertexes.append(prev[f])
                f = prev[f]
                path(prev, pathByVertexes, f)
            return pathByVertexes[::-1]
        pathList = path(prev, pathByVertexes, f)
        print('Path found. Vertexes to pass through:')
        print(*pathList)
    return pathList

pathListDjikstra = shortestPathFastDjikstra(N)
#endregion

#region A*
def adjacencyListCreation(N):

    f = open('points16.txt')
    lines = f.read().splitlines()
    print('\nNumber of edges:', len(lines))
    print('Number of vertexes: ', N)

    adjacency_list = {i+1: [] for i in range(len(lines))}

    for i in range(len(lines)):
        a, b, ll = map(float, lines[i].split())
        a, b = int(a), int(b)
        adjacency_list[a].append((b, ll))
        adjacency_list[b].append((a, ll))

    return adjacency_list

adjacency_list = adjacencyListCreation(N) # pts and x,y of start and final points

### Here goes A* algo for path planning ###
### Thanks to https://github.com/VikashPR/18CSC305J-AI/blob/main/A_Star-BFS.py for code parts

class Graph:
    # example of adjacency list (or rather map)
    # adjacency_list = {
    # 'A': [('B', 1), ('C', 3), ('D', 7)],
    # 'B': [('D', 5)],
    # 'C': [('D', 12)]
    # }

    def __init__(self, adjacency_list):
        self.adjacency_list = adjacency_list
        
    def get_neighbors(self, v):
        return self.adjacency_list[v]

    # heuristic function with equal values for all nodes
    def h(self, n, pointsAndCoordinates):

        goalDist = np.sqrt((pointsAndCoordinates[N][0]-pointsAndCoordinates[n][0])**2 + 
                           (pointsAndCoordinates[N][1]-pointsAndCoordinates[n][1])**2)

        return goalDist

    def a_star_algorithm(self, start_node, stop_node, pointsAndCoordinates):
        # open_list is a list of nodes which have been visited, but who's neighbors
        # haven't all been inspected, starts off with the start node
        # closed_list is a list of nodes which have been visited
        # and who's neighbors have been inspected
        open_list = set([start_node])
        closed_list = set([])

        # g contains current distances from start_node to all other nodes
        # the default value (if it's not found in the map) is +infinity
        g = {}

        g[start_node] = 0

        # parents contains an adjacency map of all nodes
        parents = {}
        parents[start_node] = start_node
        dist = {}

        while len(open_list) > 0:
            n = None

            # find a node with the lowest value of f() - evaluation function
            for v in open_list:
                if n == None or g[v] + self.h(v, pointsAndCoordinates) < g[n] + self.h(n, pointsAndCoordinates):
                    n = v

            if n == None:
                print('Path does not exist!')
                return None

            # if the current node is the stop_node
            # then we begin reconstructin the path from it to the start_node
            if n == stop_node:
                reconst_path = []

                while parents[n] != n:
                    reconst_path.append(n)
                    n = parents[n]

                reconst_path.append(start_node)

                reconst_path.reverse()
                print('\nDistance to cover:', round(dist[N], 3))
                print('Path found. Vertexes to pass through:')
                print(*reconst_path)
                return reconst_path

            # for all neighbors of the current node do
            for (m, weight) in self.get_neighbors(n):
                # if the current node isn't in both open_list and closed_list
                # add it to open_list and note n as it's parent
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight
                    dist[m] = g[m]
                # otherwise, check if it's quicker to first visit n, then m
                # and if it is, update parent data and g data
                # and if the node was in the closed_list, move it to open_list
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        dist[m] = g[m]

                        parents[m] = n
                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)

            # remove n from the open_list, and add it to closed_list
            # because all of his neighbors were inspected
            open_list.remove(n)
            closed_list.add(n)
        print('Path does not exist!')
        return None

graph1 = Graph(adjacency_list)
pathListAstar = graph1.a_star_algorithm(N-1, N, pointsAndCoordinates) # N-1 and N are numbers of start and final points
#endregion

def imageSaveDjikstra(kp_image, pathList, pointsAndCoordinates, obstaclesList):
    for obstacle in obstaclesList:
        kp_image = cv2.circle(kp_image, (int(obstacle[0]),
                                     int(obstacle[1])), obstacle[2], (200, 20, 20), -1)
    pointCoords = []
    for v in pathList:
        pointCoords.append(pointsAndCoordinates[v])
    pointCoords = np.array(pointCoords, dtype=int)

    for i in range(len(pointCoords) - 1):
        newimg = cv2.line(kp_image, tuple(pointCoords[i]), tuple(pointCoords[i+1]), (0,0,255), 2)
    cv2.imwrite('Images/Djikstra3Akaze2.jpg', newimg)

    print("\nExecution of Djikstra's algorithm:", str(round((time.time()-startTime), 2))+'s')

def imageSaveAstar(kp_image, pathList, pointsAndCoordinates, obstaclesList):
    for obstacle in obstaclesList:
        kp_image = cv2.circle(kp_image, (int(obstacle[0]),
                                     int(obstacle[1])), obstacle[2], (200, 20, 20), -1)
    pointCoords = []
    for v in pathList:
        pointCoords.append(pointsAndCoordinates[v])
    pointCoords = np.array(pointCoords, dtype=int)

    for i in range(len(pointCoords) - 1):
        newimg = cv2.line(kp_image, tuple(pointCoords[i]), tuple(pointCoords[i+1]), (0,0,255), 2)
    cv2.imwrite('Images/Astar3Akaze2.jpg', newimg)
    print("\nExecution of A* algorithm:", str(round((time.time()-startTime), 2))+'s')

imageSaveDjikstra(kp_image, pathListDjikstra, pointsAndCoordinates, obstaclesList)
imageSaveAstar(kp_image, pathListAstar,  pointsAndCoordinates, obstaclesList)
