const express = require('express');
const app = express();


app.get('/', (req, res) => {
  const code = `
/linear   - Linear Search
/binary   - Binary Search
/quick    - Quick Sort
/merge    - Merge Sort
/prims    - Prims Algorithm
/kruskal  - Kruskals Algorithm
/nqueens  - N'Queens Problem 
/dijkstra - Dijkstra's Algorithm
/knapsack - Knapsack Problem
/floyd    - Floyd Warshall Algorithm
/omp      - Optimal Merge Pattern
/magic    - Magic Square
/binarycpp - Binary C++
/linearcpp - Linear C++
/magic6cpp - Magic Square 6 c++
`;
  res.type('text/plain');
  res.send(code);
});

// 🟢 Linear Search
app.get('/linear', (req, res) => {
  const code = `
def linear_search(arr, key):
    for i in range(0, len(arr)):
        if arr[i] == key:
            return i + 1
    return -1


print(linear_search([1, 2, 3, 4, 5], 3))
`;
  res.type('text/plain');
  res.send(code);
});

// 🔵 Binary Search
app.get('/binary', (req, res) => {
  const code = `
def binary_search(arr, key):
    low = 0
    high = len(arr) - 1
    mid = int((low + high) / 2)

    while low <= high:
        if arr[mid] == key:
            return mid + 1
        elif arr[mid] > key:
            high = mid - 1
        elif arr[mid] < key:
            low = mid + 1
        mid = int((low + high) / 2)
    return -1


print(binary_search([1, 4, 7, 9, 10, 15], 101))
`;
  res.type('text/plain');
  res.send(code);
});

// 🟣 Merge Sort
app.get('/merge', (req, res) => {
  const code = `
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0 

    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else: 
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result


arr = [5, 3, 2, 1, 4, 5, 1, 2, 3]
sorted_arr = merge_sort(arr)
print(sorted_arr)
`;
  res.type('text/plain');
  res.send(code);
});

// 🟡 Quick Sort
app.get('/quick', (req, res) => {
  const code = `
def quick_sort(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[-1]

    left =  [x for x in arr[:-1] if x <= pivot]
    right = [x for x in arr[:-1] if x >= pivot]

    return quick_sort(left) + [pivot] + quick_sort(right)


print(quick_sort([5, 4, 3, 2, 1]))
`;
  res.type('text/plain');
  res.send(code);
});

// 🟠 Prim’s Algorithm
app.get('/prims', (req, res) => {
  const code = `
import heapq

def prims(graph, start):
    visited = set()
    min_heap = [(0, start)]  
    total_cost = 0

    while min_heap:
        weight, u = heapq.heappop(min_heap)
        
        if u in visited:
            continue
        
        visited.add(u)
        total_cost += weight

        for v, w in graph[u]:
            if v not in visited:
                heapq.heappush(min_heap, (w, v))

    return total_cost
  
  
graph = {
    'A': [('B', 2), ('C', 3)],
    'B': [('A', 2), ('C', 1), ('D', 4)],
    'C': [('A', 3), ('B', 1), ('D', 5)],
    'D': [('B', 4), ('C', 5)]
}

print("Total cost of MST using Prim's:", prims(graph, 'A'))
`;
  res.type('text/plain');
  res.send(code);
});

// 🔴 N-Queens
app.get('/nqueens', (req, res) => {
  const code = `
def print_grid(grid, row): 
    n = 4  # define the board size once

    if row == n: 
        # base case – all queens placed
        for r in grid:
            print(r)
        print()   # blank line between solutions
        return

    for i in range(n):
        if is_safe(grid, row, i, n):
            grid[row][i] = "Q"
            print_grid(grid, row + 1)
            grid[row][i] = " "  # backtrack


def is_safe(grid, row, col, n):
    # check column 
    for i in range(row): 
        if grid[i][col] == "Q":
            return False 

    # check left diagonal
    i, j = row - 1, col - 1
    while i >= 0 and j >= 0:
        if grid[i][j] == "Q":
            return False
        i -= 1
        j -= 1

    # check right diagonal
    i, j = row - 1, col + 1
    while i >= 0 and j < n:
        if grid[i][j] == "Q":
            return False
        i -= 1
        j += 1

    return True 


# initialize 4×4 empty board and start solving
print_grid([[" " for _ in range(4)] for _ in range(4)], 0)
`;
  res.type('text/plain');
  res.send(code);
});


app.get('/kruskal', (req, res) => {
  const code = `
# Kruskal's Algorithm - Simple & Memorable

def find(parent, i):
    if parent[i] != i:
        parent[i] = find(parent, parent[i])  # Path compression
    return parent[i]

def union(parent, rank, x, y):
    if rank[x] > rank[y]:
        parent[y] = x
    elif rank[x] < rank[y]:
        parent[x] = y
    else:
        parent[y] = x
        rank[x] += 1

def kruskal(graph):
    V = len(graph)  # Number of vertices
    edges = []
    
    # Collect all edges: (weight, u, v)
    for u in range(V):
        for v, weight in graph[u]:
            edges.append((weight, u, v))
    
    # Sort edges by weight
    edges.sort()
    
    parent = [i for i in range(V)]
    rank = [0] * V
    mst = []
    total_cost = 0
    
    for weight, u, v in edges:
        pu = find(parent, u)
        pv = find(parent, v)
        
        if pu != pv:  # No cycle
            union(parent, rank, pu, pv)
            mst.append((u, v, weight))
            total_cost += weight
    
    return mst, total_cost

# Example usage:
graph = [
    [(1, 2), (3, 4)],     # Node 0 connected to 1(w=2), 3(w=4)
    [(0, 2), (2, 3)],     # Node 1
    [(1, 3), (3, 5)],     # Node 2
    [(0, 4), (2, 5)]      # Node 3
]

mst, cost = kruskal(graph)
print("MST Edges:", mst)
print("Total Cost:", cost)
`;
  res.type('text/plain');
  res.send(code);
});

// 🟢 Dijkstra's Algorithm
app.get('/dijkstra', (req, res) => {
  const code = `
import heapq

def dijkstra(graph, start):
    distance = {node: float('inf') for node in graph}
    distance[start] = 0
    queue = [(0, start)]

    while queue:
        dist, node = heapq.heappop(queue)

        for neighbor, weight in graph[node]:
            if dist + weight < distance[neighbor]:
                distance[neighbor] = dist + weight
                heapq.heappush(queue, (distance[neighbor], neighbor))
    return distance

graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('A', 1), ('C', 2), ('D', 5)],
    'C': [('A', 4), ('B', 2), ('D', 1)],
    'D': [('B', 5), ('C', 1)]
}

print(dijkstra(graph, 'A'))
`;
  res.type('text/plain');
  res.send(code);
});

// 🟢 Knapsack Problem
app.get('/knapsack', (req, res) => {
  const code = `
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i-1][w], values[i-1] + dp[i-1][w - weights[i-1]])
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][capacity]

weights = [1, 2, 3, 4]
values = [10, 20, 30, 40]
capacity = 5
print(knapsack(weights, values, capacity))
`;
  res.type('text/plain');
  res.send(code);
});

// 🟢 Floyd Warshall Algorithm
app.get('/floyd', (req, res) => {
  const code = `
def floyd_warshall(graph):
    n = len(graph)
    dist = [row[:] for row in graph]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist

graph = [
    [0, 3, float('inf'), 7],
    [8, 0, 2, float('inf')],
    [5, float('inf'), 0, 1],
    [2, float('inf'), float('inf'), 0]
]

res = floyd_warshall(graph)
for row in res:
    print(row)
`;
  res.type('text/plain');
  res.send(code);
});

// 🟢 Optimal Merge Pattern
app.get('/omp', (req, res) => {
  const code = `
import heapq

def optimal_merge(files):
    heapq.heapify(files)
    cost = 0
    while len(files) > 1:
        first = heapq.heappop(files)
        second = heapq.heappop(files)
        merged = first + second
        cost += merged
        heapq.heappush(files, merged)
    return cost

files = [2, 3, 7, 10]
print(optimal_merge(files))
`;
  res.type('text/plain');
  res.send(code);
});

// 🟢 Magic Square Generation (Odd Order)
app.get('/magic', (req, res) => {
  const code = `
def generate_magic_square(n):
    magic = [[0] * n for _ in range(n)]
    i, j = 0, n // 2
    for num in range(1, n * n + 1):
        magic[i][j] = num
        ni, nj = (i - 1) % n, (j + 1) % n
        if magic[ni][nj]:
            i = (i + 1) % n
        else:
            i, j = ni, nj
    for row in magic:
        print(row)

generate_magic_square(3)
`;
  res.type('text/plain');
  res.send(code);
});



app.get('/binarycpp', (req, res) => {
  const code = `
  #include <bits/stdc++.h>
using namespace std;

string randomStr(int len) {
    static const string chars = "abcdefghijklmnopqrstuvwxyz";
    string s;
    for(int i = 0; i < len; i++)
        s += chars[rand() % chars.size()];
    return s;
}

int binaryRec(vector<string>& a, string key, int l, int r) {
    if(l > r) return -1;
    int mid = (l+r)/2;
    if(a[mid] == key) return mid;
    if(a[mid] < key) return binaryRec(a, key, mid+1, r);
    return binaryRec(a, key, l, mid-1);
}

int main() {
    srand(time(0));
    int n;
    cin >> n;
    vector<string> a(n);
    for(int i = 0; i < n; i++)
        a[i] = randomStr(5);

    sort(a.begin(), a.end());

    string key;
    cin >> key;

    int idx = binaryRec(a, key, 0, n-1);
    if(idx == -1) cout << "Not found";
    else cout << "Found at index " << idx;
}
`;
  res.type('text/plain');
  res.send(code);
});



app.get('/linearcpp', (req, res) => {
  const code = `
  #include <bits/stdc++.h>
using namespace std;

string randomStr(int len) {
    static const string chars = "abcdefghijklmnopqrstuvwxyz";
    string s;
    for(int i = 0; i < len; i++)
        s += chars[rand() % chars.size()];
    return s;
}

int linearRec(vector<string>& a, string key, int i) {
    if(i == a.size()) return -1;
    if(a[i] == key) return i;
    return linearRec(a, key, i+1);
}

int main() {
    srand(time(0));
    int n;
    cin >> n;
    vector<string> a(n);
    for(int i = 0; i < n; i++)
        a[i] = randomStr(5);

    string key;
    cin >> key;

    int idx = linearRec(a, key, 0);
    if(idx == -1) cout << "Not found";
    else cout << "Found at index " << idx;
}
`;
  res.type('text/plain');
  res.send(code);
});

app.get('/magic6cpp', (req, res) => {
  const code = `
#include <iostream>
using namespace std;

int main() {
    int n = 6;
    int square[6][6];

    int n2 = n / 2;
    int subSquare[3][3];
    int k = 1;

    for(int i = 0; i < n2; i++)
        for(int j = 0; j < n2; j++)
            subSquare[i][j] = k++;

    int offset = n2 * n2;

    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            square[i][j] = subSquare[i % n2][j % n2];

    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++) {
            if (i < n2 && j < n2) square[i][j] += 0 * offset;
            else if (i < n2 && j >= n2) square[i][j] += 2 * offset;
            else if (i >= n2 && j < n2) square[i][j] += 3 * offset;
            else square[i][j] += 1 * offset;
        }

    int m = (n - 2) / 4;

    for(int i = 0; i < n; i++)
        for(int j = 0; j < m; j++)
            swap(square[i][j], square[i + n2][j]);

    for(int i = 0; i < n; i++)
        for(int j = n - m + 1; j < n; j++)
            swap(square[i][j], square[i + n2][j]);

    swap(square[m][0], square[m + n2][0]);
    swap(square[m][m], square[m + n2][m]);

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++)
            cout << square[i][j] << "\t";
        cout << endl;
    }
}
`;
  res.type('text/plain');
  res.send(code);
});


app.get('/a', (req, res) => {
  const code = `
`;
  res.type('text/plain');
  res.send(code);
});


app.listen(46750, () => {
  console.log("chalu thai gyu");
});
