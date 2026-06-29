from collections import deque


def topological_sort(nodes, edges):
    in_edges  = {n: [] for n in nodes}
    out_edges = {n: [] for n in nodes}
    for e in edges:
        in_edges[e["dst"]].append(e)
        out_edges[e["src"]].append(e)

    in_degree = {n: len(in_edges[n]) for n in nodes}
    queue     = deque(n for n, d in in_degree.items() if d == 0)
    order     = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for e in out_edges[node]:
            in_degree[e["dst"]] -= 1
            if in_degree[e["dst"]] == 0:
                queue.append(e["dst"])
    return order
