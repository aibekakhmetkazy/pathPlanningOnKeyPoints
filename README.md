The project is related to solve path planning task using A-star, Bidirectional A*, RRT-Connect and fast Dijkstra's algorithm with priority queues on satellite images.

Here you can see how different algorithms performed on an satellite image:

Number of vertexes: 10611

Number of edges: 324456 

**A-star** algorithm performance:

<img src="/GIFs/A*.gif" width="80%">

A* found a path:

 [10610, 1022, 1329, 1636, 1943, 2250, 2455, 2865, 3275, 3581, 3783, 3883, 4085, 4391, 4698, 4901, 5103, 10611]

Total distance: 329.37

------------------------------------------------
**Dijkstra's** algorithm performance:

<img src="/GIFs/Dijkstra.gif" width="80%">

Dijkstra's found a path:

 [10610, 920, 1227, 1534, 1841, 2148, 2455, 2865, 3275, 3581, 3681, 3883, 4085, 4392, 4698, 4901, 5103, 10611]

Total distance: 329.37

------------------------------------------------
**Bi-A-star** algorithm performance:

<img src="/GIFs/Bi-A*.gif" width="80%">

Bidirectional A* found a path:

 [10610, 1021, 1327, 1634, 1837, 2040, 2242, 2444, 2647, 2954, 3367, 3674, 3879, 4084, 4391, 4698, 4901, 5002, 5103, 10611]

Total distance: 336.689

------------------------------------------------
**RRT-connect** algorithm performance:

<img src="/GIFs/RRTConnect.gif" width="80%">

d-RRT Connect found a path:

 [10610, 608, 1018, 1429, 1839, 2249, 2451, 2241, 2443, 2749, 3159, 3571, 3883, 3675, 3877, 4287, 4699, 4901, 5000, 10611]

Total distance: 406.861

------------------------------------------------

**A-star** algorithm performance with **SIFT** key points extractor:

<img src="/Images/AstarSift16.png" width="80%">

**A-star** algorithm performance with **AKAZE** key points extractor:

<img src="/Images/AstarAkaze16.png" width="80%">

**Dijkstra's** algorithm performance with **SIFT** key points extractor:

<img src="/Images/DijkstraSift16.png" width="80%">

**Dijkstra's** algorithm performance with **AKAZE** key points extractor:

<img src="/Images/DijkstraAkaze16.png" width="80%">