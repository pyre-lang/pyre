# pyre-check: max-pypy-ratio=10
# Dijkstra over a 150 x 150 weighted grid with `heapq`; each heap entry is one
# int (`d * N*N + cell`) and `divmod` splits it back.
#
# PENDING (do not move to pyre/bench/synth/ until it meets the gate):
# `run_synthetic_suite` globs `*.py` non-recursively, so the gate skips this
# file. Measured 2026-10-29 on 74c40239cb5 (macOS arm64): pyre-dynasm 1.68 s
# with loops_compiled=5, bridges_compiled=103, guard_failures=38678; the web
# wasm build takes 6.6 s. pypy3 3.11 runs it in 0.05 s (7 loops, 16 bridges,
# no aborts) and CPython in 0.04 s. Issue #2514.
import heapq

N = 150


def shortest(cost, sx, sy, tx, ty):
    NN = N * N
    dist = [1 << 60] * NN
    dist[sy * N + sx] = 0
    heap = [sy * N + sx]
    while heap:
        d, cell = divmod(heapq.heappop(heap), NN)
        if d > dist[cell]:
            continue
        if cell == ty * N + tx:
            return d
        y, x = divmod(cell, N)
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < N and 0 <= ny < N:
                new_dist = d + cost[ny][nx]
                ncell = ny * N + nx
                if new_dist < dist[ncell]:
                    dist[ncell] = new_dist
                    heapq.heappush(heap, new_dist * NN + ncell)
    return -1


def main():
    cost = [[(x * 37 + y * 91 + x * y) % 9 + 1 for x in range(N)] for y in range(N)]
    print(shortest(cost, 0, 0, N - 1, N - 1))
    print(shortest(cost, N // 2, N // 2, 0, N // 2))


main()
