const express = require('express');
const app = express();

app.get('/', (req, res) => {
  res.send("tari ben no piko");
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

app.listen(46750, () => {
  console.log("chalu thai gyu");
});

