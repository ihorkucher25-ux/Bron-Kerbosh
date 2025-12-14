import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Set, List
import random
from itertools import combinations
from networkx.classes import graph
import time

def draw_graph(graph):
    G = nx.Graph()
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    pos = nx.spring_layout(G)
    nx.draw(G,pos, with_labels=True, node_color="gray",
            node_size=800, font_size=12, edge_color="black")
    plt.show(block=False)
    plt.pause(1)

def bron_kerbosh(graph: Dict[int, Set[int]]) -> List[Set[int]]:
    cliques: List[Set[int]] = []
    def bk(R: Set[int], P: Set[int], X: Set[int]) -> None:
        if not P and not X:
            cliques.append(set(R))
            return
        for v in list(P):
            N_v = graph.get(v, set())
            bk(R | {v}, P & N_v, X & N_v)
            P.remove(v)
            X.add(v)
    all_vertices = set(graph.keys())
    bk(set(), set(all_vertices), set())
    return cliques


def adjacency_matrix_to_dict(matrix):
    n = len(matrix)
    graph = {}
    for i in range(n):
        neighbors = set()
        for j in range(n):
            if matrix[i][j] == 1:
                neighbors.add(j + 1)
        graph[i + 1] = neighbors
    return graph


def draw_clique(nodes):
    plt.figure()

    G = nx.Graph()
    nodes_list = list(nodes)
    G.add_nodes_from(nodes_list)
    for i in range(len(nodes_list)):
        for j in range(i + 1, len(nodes_list)):
            G.add_edge(nodes_list[i], nodes_list[j])

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="lightblue",
            node_size=800, font_size=12, edge_color="black")
    plt.show(block=False)
    plt.pause(0.05)


def random_matrix(n: int, density: float, seed: int):
    if not (0 <= density <= 1):
        raise ValueError("Щільність має бути у межах [0,1]")
    if seed is not None:
        random.seed(seed)
    max_edges = n * (n - 1) // 2
    m = int(round(density * max_edges))
    possible_edges = list(combinations(range(n), 2))
    chosen_edges = random.sample(possible_edges, m)
    A = [[0] * n for i in range(n)]
    for u, v in chosen_edges:
        A[u][v] = 1
        A[v][u] = 1
    return A

# Виконання і графічне представлення алгоритму
# n=int(input("Введіть кількість вершин:"))
# d=float(input("Вкажіть щільність(від 0 до 1):"))
# A=random_matrix(n, d,None)
# B=adjacency_matrix_to_dict(A)
# C=bron_kerbosh(B)
# print(f'Graph: {B} has cliques: {C}')
# draw_graph(B)
# plt.pause(1)
# for i in C:
#     draw_clique(i)
#     plt.pause(1)
# plt.pause(600)


def experiment(sizes, density, seed=None):
    results = []
    for n in sizes:
        A = random_matrix(n, density, seed)
        B = adjacency_matrix_to_dict(A)

        start = time.perf_counter()
        C = bron_kerbosh(B)
        end = time.perf_counter()

        elapsed = end - start
        results.append((n, elapsed, len(C)))
        print(f"n={n}, cliques={len(C)}, time={elapsed:.6f} sec")
    return results

#Дослідження часу виконання
# sizes=[]
# for i in range(5,100,15):
#     sizes.append(i)
# for i in range(100,350,25):
#     sizes.append(i)
# sizes.append(500)
# density = 0.3
# results = experiment(sizes, density, seed=42)
#
# x = [r[0] for r in results]
# y = [r[1] for r in results]
# plt.plot(x, y, marker="o")
# plt.xlabel("Кількість вершин (n)")
# plt.ylabel("Час виконання (секунди)")
# plt.title("Час роботи алгоритму Брона–Кербоша")
# plt.grid(True)
# plt.show()








