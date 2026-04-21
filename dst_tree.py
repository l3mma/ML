import math

adjacency_matrix = [
    [0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
    [1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0]
]

n = 12
c = [[float('inf')] * n for _ in range(n)]

# c_ij = (i+j) % 3
for i in range(n):
    for j in range(n):
        if adjacency_matrix[i][j] == 1:
            weight = ((i+1) + (j+1)) % 3
            c[i][j] = weight

print("Матрица смежности икосаэдра")
print("   ", end="")
for i in range(1, 13):
    print(f"{i:3}", end="")
print()

for i in range(12):
    print(f"{i+1:2}:", end="")
    for j in range(12):
        if c[i][j] == float('inf'):
            print("  ∞", end="")
        else:
            print(f"{int(c[i][j]):3}", end="")
    print()

def prim_algorithm(c):
    n = len(c)
    V = set(range(1, n+1))
    T = []
    s = 1
    V = V - {s}
    d = [float('inf')] * (n+1)
    p = [-1] * (n+1)
    d[s] = 0
    
    for v in V:
        d[v] = c[s-1][v-1]
        p[v] = s
    
    iteration = 1
    
    while V:
        print(f"\nИТЕРАЦИЯ {iteration}")
        
        min_d = float('inf')
        u = -1
        for v in V:
            if d[v] < min_d:
                min_d = d[v]
                u = v
        
        print(f"Выбрана вершина u = {u} с d[u] = {d[u]}")
        print(f"T = T ∪ {{{p[u]}-{u}}}")
        
        V = V - {u}
        T.append((p[u], u))
        
        print(f"Добавлено ребро: {p[u]} - {u} (вес: {c[p[u]-1][u-1]})")
        
        updated = False
        for v in V:
            if c[u-1][v-1] < d[v]:
                print(f"Обновляем вершину {v}: c[{u}][{v}] = {c[u-1][v-1]} < d[{v}] = {d[v]}")
                d[v] = c[u-1][v-1]
                p[v] = u
                updated = True
        
        if not updated:
            print("Нет обновлений меток на этой итерации")
        
        iteration += 1
    
    return T

minimum_spanning_tree = prim_algorithm(c)

print("\n" + "="*50)
print("РЕЗУЛЬТАТ")
print("="*50)
print("Остовное дерево минимального веса (рёбра):")
for edge in minimum_spanning_tree:
    print(f"{edge[0]} - {edge[1]}")

total_weight = sum(c[u-1][v-1] for u, v in minimum_spanning_tree)
print(f"Общий вес остовного дерева: {total_weight}")