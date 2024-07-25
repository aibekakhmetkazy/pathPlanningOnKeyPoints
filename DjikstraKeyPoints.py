import cv2
import heapq
import numpy as np
import matplotlib.pyplot as plt
import time

startTime = time.time()
img = cv2.imread('high_resolution_image17.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initiate SIFT object with default values
# sift = cv2.SIFT_create()
# # find the keypoints on image (grayscale)
# kp = sift.detect(gray, None)
# # draw keypoints in image
# img2 = cv2.drawKeypoints(gray, kp, None, flags=0)


# # Applying the function 
# orb = cv2.ORB_create(nfeatures=2000) 
# kp, des = orb.detectAndCompute(gray, None) 
# # Drawing the keypoints 
# kp_image = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0) 


# fast = cv2.FastFeatureDetector_create() 
# fast.setNonmaxSuppression(False) 
# # # Drawing the keypoints 
# kp = fast.detect(gray, None) 
# kp_image = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0)) 


akaze = cv2.AKAZE_create()
keypoints, descriptors = akaze.detectAndCompute(img, None)
kp_image = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)

# cv2.imwrite('akaze.jpg', kp_image)

pts = cv2.KeyPoint_convert(keypoints)
x, y = pts[:, 0], pts[:, 1]

### Here goes Djikstra algo for path planning ###

def shortestPathFastDjikstra(pts, startx, starty, finalx, finaly):

    pointsAndCoordinates = [[]]
    n = len(pts[:, 0])

    for i in range(1, n+1):
        pointsAndCoordinates.append([pts[i-1, 0], pts[i-1, 1]])

    pointsAndCoordinates.append([startx, starty])   # Start vertex is n
    pointsAndCoordinates.append([finalx, finaly])   # Final vertex is n+1

    N = n+2     # Number of vertexes increased after addition of start and final vertexes

    obstaclesList = [[550, 550, 30], [700, 400, 50]]  # List of given circle obstacles with x,y,r values
    obstacles = []

    minRad = float('inf')

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
    # adjList = []
    # fileStart = time.time()
    # with open('points.txt', 'w') as f:  # Writing all the vertexes and distances into 1 txt file for matrix construction
    #     for i in range(1, N):
    #         for j in range(i+1, N+1):
    #             dist = round(((pointsAndCoordinates[i][0] - pointsAndCoordinates[j][0])**2 +
    #                     (pointsAndCoordinates[i][1] - pointsAndCoordinates[j][1])**2)**.5, 2)
    #             if dist < minRad and i not in obstacles and j not in obstacles:
    #                 adjList.append([i, j, dist])
    #                 # f.write("{} {} {}\n".format(i, j, dist))
    #     f.write("\n".join([" ".join(map(str, a)) for a in adjList]))
    # print('points.txt file creation took:',round(time.time()-fileStart, 2))
    #endregion

    f = open('points.txt')
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
    return pathList, pointsAndCoordinates, obstaclesList

pathList, pointsAndCoordinates, obstaclesList = shortestPathFastDjikstra(pts, 60, 60, 700, 700)

for obstacle in obstaclesList:
    kp_image = cv2.circle(kp_image, (int(obstacle[0]),
                                     int(obstacle[1])), obstacle[2], (0, 0, 0), -1)

pointCoords = []
for v in pathList:
    pointCoords.append(pointsAndCoordinates[v])
pointCoords = np.array(pointCoords, dtype=int)

for i in range(len(pointCoords) - 1):
    newimg = cv2.line(kp_image, tuple(pointCoords[i]), tuple(pointCoords[i+1]), (0,0,255), 2)
cv2.imwrite('DjikstraWithObstacleOnTifImage.jpg', newimg)

print("\nExecution of Djikstra's algorithm:", str(round((time.time()-startTime), 2))+'s')