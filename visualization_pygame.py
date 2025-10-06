# visualize_your_sorts.py
import random, pygame, sys, math

# ---------- Window & style ----------
W, H = 1000, 560
PADDING_X, PADDING_Y = 50, 60
N_BARS = 140
FPS_CAP = 240

BG = (16, 18, 22)
BAR = (180, 200, 255)
CUR = (255, 120, 120)
AUX = (255, 210, 120)
DONE = (120, 220, 160)
TEXT = (210, 212, 220)

pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Sorting Visualizer")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 18)

# ---------- Data helpers ----------
def make_int_data(n):
    # integers 1..n shuffled
    a = list(range(1, n + 1))
    random.shuffle(a)
    return a

def make_float01_data(n):
    # floats in [0,1); mild clustering for nicer buckets visuals
    return [min(0.9999, max(0.0, random.random() ** random.uniform(0.6, 1.4))) for _ in range(n)]

# ---------- Drawing ----------
def draw_bars(arr, highlight=(), finished=False, title="", info_lines=()):
    screen.fill(BG)
    n = len(arr)
    if n == 0:
        return
    # auto scale (works for ints or floats)
    max_val = max(arr) if arr else 1
    min_val = min(arr) if arr else 0
    span = max(1e-9, max_val - min_val)
    w = (W - 2 * PADDING_X) / n
    for i, v in enumerate(arr):
        x = PADDING_X + i * w
        # normalize to [0,1]
        h_norm = (v - min_val) / span
        h = h_norm * (H - 2 * PADDING_Y)
        y = H - PADDING_Y - h
        color = DONE if finished else (CUR if i in highlight else BAR)
        pygame.draw.rect(screen, color, (x, y, max(1, int(w) - 1), int(h)))
    # UI text
    ytxt = 10
    for line in ([title] + list(info_lines)):
        surf = font.render(line, True, TEXT)
        screen.blit(surf, (10, ytxt))
        ytxt += 22

#Sorting algorithm generators

# Bubble Sort 
def bubble_sort_gen(a, stats):
    n = len(a)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            stats["comparisons"] += 1
            yield (j, j + 1)
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
                stats["swaps"] += 1
                yield (j, j + 1)
        if not swapped:
            break

# Insertion Sort 
def insertion_sort_gen(a, stats):
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0:
            stats["comparisons"] += 1
            yield (j, i)
            if key < a[j]:
                a[j + 1] = a[j]
                stats["swaps"] += 1
                yield (j, j + 1)
                j -= 1
            else:
                break
        a[j + 1] = key
        yield (j + 1, i)

# Merge Sort (top-down) with stable merge like your merge(left,right)  [turn0file6]
def merge_sort_gen(a, stats):
    
    n = len(a)
    width = 1
    aux = [0] * n
    while width < n:
        for left in range(0, n, 2 * width):
            mid = min(left + width, n)
            right = min(left + 2 * width, n)
            i, j, k = left, mid, left
            while i < mid and j < right:
                stats["comparisons"] += 1
                yield (i, j)
                if a[i] <= a[j]:
                    aux[k] = a[i]; i += 1
                else:
                    aux[k] = a[j]; j += 1
                k += 1
            while i < mid:
                aux[k] = a[i]; i += 1; k += 1; yield (i - 1,)
            while j < right:
                aux[k] = a[j]; j += 1; k += 1; yield (j - 1,)
            for k2 in range(left, right):
                a[k2] = aux[k2]
                yield (k2,)
        width *= 2

# Quick Sort 
def quick_sort_gen(a, stats):
    def partition(lo, hi):
        pivot = a[(lo + hi) // 2]
        i, j = lo, hi
        while i <= j:
            while a[i] < pivot:
                stats["comparisons"] += 1
                yield (i,)
                i += 1
            stats["comparisons"] += 1
            while a[j] > pivot:
                stats["comparisons"] += 1
                yield (j,)
                j -= 1
            stats["comparisons"] += 1
            if i <= j:
                a[i], a[j] = a[j], a[i]
                stats["swaps"] += 1
                yield (i, j)
                i += 1; j -= 1
        return i, j

    stack = [(0, len(a) - 1)]
    while stack:
        lo, hi = stack.pop()
        if lo >= hi: 
            continue
        # run partition as subgen
        part = partition(lo, hi)
        try:
            while True:
                h = next(part)
                yield h
        except StopIteration as stop:
            i, j = stop.value if stop.value else (lo, hi)
        # Recursively sort segments
        if lo < j: stack.append((lo, j))
        if i < hi: stack.append((i, hi))

# Heap Sort (build_max_heap + sift_down)  [turn0file4]
def heap_sort_gen(a, stats):
    n = len(a)
    def sift_down(start, end):
        root = start
        while True:
            left = 2 * root + 1
            if left > end:
                break
            right = left + 1
            largest = root
            # compare left
            stats["comparisons"] += 1
            if a[left] > a[largest]:
                largest = left
            # compare right
            if right <= end:
                stats["comparisons"] += 1
                if a[right] > a[largest]:
                    largest = right
            if largest == root:
                break
            a[root], a[largest] = a[largest], a[root]
            stats["swaps"] += 1
            yield (root, largest)
            root = largest

    # build max-heap
    for i in range(n // 2 - 1, -1, -1):
        s = sift_down(i, n - 1)
        for h in s: 
            yield h
    # pull-down
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        stats["swaps"] += 1
        yield (0, end)
        s = sift_down(0, end - 1)
        for h in s: 
            yield h

# Counting Sort
def counting_sort_gen(a, stats):
    if not a:
        return
    min_val, max_val = min(a), max(a)
    k = max_val - min_val + 1
    count = [0] * k
    # ount
    for v in a:
        count[v - min_val] += 1
        yield ()
    # prefix sums
    for i in range(1, k):
        count[i] += count[i - 1]
        yield ()
    # output (stable, iterate backwards)
    out = [0] * len(a)
    for i in range(len(a) - 1, -1, -1):
        idx = a[i] - min_val
        count[idx] -= 1
        out[count[idx]] = a[i]
        yield (i,)
    # copy back
    for i in range(len(a)):
        a[i] = out[i]
        yield (i,)

# Bucket Sort
def bucket_sort_gen(a, stats):
    n = len(a)
    if n == 0:
        return
    # Expect values in [0,1)
    amin, amax = min(a), max(a)
    norm = False
    if amin < 0 or amax >= 1.0:
        norm = True
        rng = max(1e-9, amax - amin)
        a[:] = [(x - amin) / rng for x in a]
    buckets = [[] for _ in range(n)]
    for idx, x in enumerate(a):
        bi = min(n - 1, int(n * x))
        buckets[bi].append(x)
        yield (idx,)
    for b in buckets:
        b.sort()
        yield ()
    # flatten
    k = 0
    for b in buckets:
        for v in b:
            a[k] = v; k += 1
            yield (k - 1,)
    # denormalize back if needed
    if norm:
        for i in range(n):
            a[i] = a[i] * (amax - amin) + amin
            yield (i,)

# Radix Sort
def radix_sort_lsd_gen(a, stats, base=10):
    if not a:
        return
    neg = [-x for x in a if x < 0]
    pos = [x for x in a if x >= 0]
    def nonneg_passes(arr):
        if not arr:
            return
        max_val = max(arr)
        exp = 1
        while max_val // exp > 0:
            # counting by digit
            count = [0] * base
            out = [0] * len(arr)
            for x in arr:
                d = (x // exp) % base
                count[d] += 1
                yield ()
            for d in range(1, base):
                count[d] += count[d - 1]
                yield ()
            for i in range(len(arr) - 1, -1, -1):
                d = (arr[i] // exp) % base
                count[d] -= 1
                out[count[d]] = arr[i]
                yield (i,)
            for i in range(len(arr)):
                arr[i] = out[i]
                yield (i,)
            exp *= base
    # sort neg and pos parts
    for step in nonneg_passes(neg): 
        yield step
    for step in nonneg_passes(pos): 
        yield step
    # negatives reversed & sign restored, then positives
    neg_sorted = [-x for x in reversed(neg)]
    k = 0
    for v in neg_sorted:
        a[k] = v; k += 1; yield (k - 1,)
    for v in pos:
        a[k] = v; k += 1; yield (k - 1,)

# List of algorithms
ALGOS = [
    ("Bubble", bubble_sort_gen),
    ("Insertion", insertion_sort_gen),
    ("Merge", merge_sort_gen),
    ("Quick", quick_sort_gen),
    ("Heap", heap_sort_gen),
    ("Counting", counting_sort_gen),
    ("Bucket(0..1)", bucket_sort_gen),
    ("Radix(LSD)", radix_sort_lsd_gen),
]

# ---------- Main Loop ----------
def main():
    idx = 0
    algo_name, algo_fn = ALGOS[idx]
    arr = make_int_data(N_BARS)
    stats = {"comparisons": 0, "swaps": 0}
    gen = None
    running = False
    finished = False
    speed = 1.0
    step_accum = 0.0
    steps_done = 0
    highlight = ()

    def reset_arr(for_algo):
        # Counting/Radix need ints (support negatives for Radix), Bucket needs floats ~[0,1)
        if "Bucket" in for_algo:
            return make_float01_data(N_BARS)
        elif "Radix" in for_algo:
            vals = [random.randint(-999, 999) for _ in range(N_BARS)]
            random.shuffle(vals)
            return vals
        else:
            return make_int_data(N_BARS)

    def reset_algo():
        nonlocal stats, gen, running, finished, steps_done, highlight
        stats = {"comparisons": 0, "swaps": 0}
        gen = algo_fn(arr, stats) if "Radix" not in algo_name else algo_fn(arr, stats, 10)
        running = False
        finished = False
        steps_done = 0
        highlight = ()

    arr = reset_arr(algo_name)
    reset_algo()

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_a:
                    idx = (idx - 1) % len(ALGOS)
                    algo_name, algo_fn = ALGOS[idx]
                    arr = reset_arr(algo_name)
                    reset_algo()
                    binary_mode = False
                elif e.key == pygame.K_d:
                    idx = (idx + 1) % len(ALGOS)
                    algo_name, algo_fn = ALGOS[idx]
                    arr = reset_arr(algo_name)
                    reset_algo()
                    binary_mode = False
                elif e.key == pygame.K_r:
                    arr = reset_arr(algo_name)
                    reset_algo()
                elif e.key == pygame.K_SPACE:
                    if not finished:
                        running = not running
                elif e.key == pygame.K_s:
                    if not finished:
                        try:
                            highlight = next(gen)
                            steps_done += 1
                        except StopIteration:
                            finished = True
                            highlight = ()
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed = min(50.0, speed + 0.5)
                elif e.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                    speed = max(0.5, speed - 0.5)
                
        # Update
        if not binary_mode and running and not finished:
            step_accum += speed
            while step_accum >= 1.0 and not finished:
                step_accum -= 1.0
                try:
                    highlight = next(gen)
                    steps_done += 1
                except StopIteration:
                    finished = True
                    highlight = ()

        # Draw

                try:
                    step = next(bs_gen)
                    if step[0] == "probe":
                        _, L, M, R = step
                        draw_bars(arr, highlight=(L, M, R), finished=True,
                                  title=title,
                                  info_lines=[f"Target={bs_target} | Window: L={L}, M={M}, R={R}"])
                    elif step[0] == "found":
                        _, M = step
                        draw_bars(arr, highlight=(M,), finished=True, title=title,
                                  info_lines=[f"Target={bs_target} FOUND at index {M}"])
                        bs_gen = None
                    else:
                        draw_bars(arr, highlight=(), finished=True, title=title,
                                  info_lines=[f"Target={bs_target} NOT FOUND"])
                        bs_gen = None
                except StopIteration:
                    bs_gen = None
        else:
            draw_bars(
                arr,
                highlight=tuple(highlight) if isinstance(highlight, (tuple, list)) else (),
                finished=finished,
                title=f"[{algo_name}]  (A/D) switch  (R) shuffle  (SPACE) run/pause  (S) step  (+/-) speed",
                info_lines=[f"Speed: {speed:.2f} steps/frame   Steps: {steps_done}   Comp: {stats['comparisons']}   Swaps: {stats['swaps']}"],
            )

        pygame.display.flip()
        clock.tick(FPS_CAP)

if __name__ == "__main__":
    main()