import heapq

#region Dijkstra
def shortestPathFastDijkstra(adjacency_list, N):

    m = adjacency_list.copy()
    prev = [-1]*(N+1)
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
    pathList = []
    if dist[f][0] == float('inf'):
        print(-1)
        print('There is no path to the final vertex!')
    else:
        def path(prev, pathByVertexes, f):
            if prev[f] > -1:
                pathByVertexes.append(prev[f])
                f = prev[f]
                path(prev, pathByVertexes, f)
            return pathByVertexes[::-1]
        pathList = path(prev, pathByVertexes, f)
        print("Dijkstra's found a path:\n", pathList, end='\n')
        print('Total distance:', round(dist[f][0], 3))
        print('------------------------------------------------')
    return pathList
#endregion