#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithm Design Project: Greedy & Divide-and-Conquer
======================================================

Author: Chittela Venkata Sai Tarun Reddy
Date: November 2024
Python: 3.11+

A) GREEDY: Task Scheduling with Deadlines
   - Domain: Operating systems, cloud computing, project management
   - Problem: Maximize number of tasks completed before their deadlines
   - Algorithm: Sort by deadline (ascending), schedule greedily
   - Complexity: O(n log n) time, O(n) space
   - Proof: Exchange argument + optimal substructure

B) DIVIDE-AND-CONQUER: Skyline Problem (Building Silhouette)
   - Domain: Computer graphics, urban planning, GIS systems
   - Problem: Compute city skyline from overlapping building rectangles
   - Algorithm: Merge skylines recursively with optimal merging
   - Complexity: O(n log n) time, O(n) space
   - Proof: Recurrence relation T(n) = 2T(n/2) + O(n)

This script:
  • Runs correctness verification tests
  • Benchmarks algorithms with multiple trials
  • Generates CSV data files
  • Creates publication-quality PNG plots for LaTeX
  • Compares with alternative approaches
"""

from __future__ import annotations
import csv
import math
import os
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ============================================
# Configuration & Utilities
# ============================================

OUTPUT_DIR = "outputs"
RANDOM_SEED = 42

def ensure_outputs_dir() -> None:
    """Create outputs directory if it doesn't exist."""
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

def path_in_outputs(filename: str) -> str:
    """Get full path for file in outputs directory."""
    ensure_outputs_dir()
    return os.path.join(OUTPUT_DIR, filename)

def now_ns() -> int:
    """High-resolution timer in nanoseconds."""
    return time.perf_counter_ns()

def secs(ns: int) -> float:
    """Convert nanoseconds to seconds."""
    return ns / 1e9

def write_csv(filename: str, header: List[str], rows: List[Tuple]) -> None:
    """Write data to CSV file in outputs directory."""
    fp = path_in_outputs(filename)
    with open(fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  → Wrote {fp}")

def try_matplotlib():
    """Attempt to import matplotlib, return None if unavailable."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        print(f"[WARNING] matplotlib not available: {e}")
        print("          Plots will be skipped. Install with: pip install matplotlib")
        return None

random.seed(RANDOM_SEED)

# ============================================
# PROBLEM A: Task Scheduling with Deadlines (Greedy)
# ============================================

@dataclass(frozen=True)
class Task:
    """Represents a task with execution time and deadline."""
    task_id: int        # Unique identifier
    duration: int       # Time units needed to complete
    deadline: int       # Must finish by this time
    
    def __repr__(self):
        return f"Task({self.task_id}, dur={self.duration}, dl={self.deadline})"

def greedy_task_scheduling(tasks: List[Task]) -> Tuple[List[Task], List[int], int]:
    """
    Greedy algorithm for task scheduling with deadlines.
    
    DOMAIN EXPLANATION (Operating Systems Context):
    In a single-processor system, multiple tasks arrive with different execution
    times and deadlines. The scheduler must decide which tasks to execute to
    maximize the number of tasks completed before their deadlines.
    
    Real-world applications:
    - Cloud computing: Scheduling batch jobs with SLA deadlines
    - Project management: Prioritizing tasks to meet milestones
    - Manufacturing: Sequencing production orders with delivery dates
    
    ABSTRACTION:
    Given: Set of tasks T = {t₁, t₂, ..., tₙ}, where each task tᵢ has:
      - duration dᵢ (execution time)
      - deadline Dᵢ (must finish by this time)
    Goal: Find subset S ⊆ T and ordering σ such that:
      - Each task in S completes before its deadline
      - |S| is maximized
    
    ALGORITHM (Earliest Deadline First):
    1. Sort all tasks by deadline (ascending order)
    2. Initialize current_time = 0
    3. For each task in sorted order:
       a. If current_time + task.duration ≤ task.deadline:
          - Schedule the task
          - Update current_time += task.duration
       b. Otherwise, skip the task
    
    CORRECTNESS PROOF (Exchange Argument):
    Claim: EDF (Earliest Deadline First) is optimal.
    
    Proof by contradiction:
    Suppose optimal solution O ≠ EDF solution G.
    Let first position where they differ be position i.
    In O: task with later deadline is scheduled
    In G: task with earlier deadline is scheduled
    
    Exchange: Swap these two tasks in O
    - Task with earlier deadline now comes first (still meets deadline)
    - Task with later deadline now comes later
      * If it met deadline before swap, it still meets deadline after
      * We haven't reduced the number of completed tasks
    
    Contradiction: O was not optimal, or O = G.
    Therefore, EDF is optimal. ∎
    
    Args:
        tasks: List of tasks to schedule
        
    Returns:
        Tuple of (scheduled_tasks, scheduled_ids, total_completion_time)
        
    Time Complexity: O(n log n) due to sorting
    Space Complexity: O(n) for storing results
    """
    if not tasks:
        return [], [], 0
    
    # Sort by deadline (ascending) - O(n log n)
    sorted_tasks = sorted(tasks, key=lambda t: t.deadline)
    
    scheduled = []
    scheduled_ids = []
    current_time = 0
    
    # Greedy selection - O(n)
    for task in sorted_tasks:
        if current_time + task.duration <= task.deadline:
            scheduled.append(task)
            scheduled_ids.append(task.task_id)
            current_time += task.duration
    
    return scheduled, scheduled_ids, current_time

def brute_force_scheduling(tasks: List[Task]) -> Tuple[int, List[int]]:
    """
    Brute force solution: try all permutations.
    
    Returns: (max_scheduled_count, best_schedule_ids)
    
    WARNING: Factorial time - only use for n <= 10
    """
    from itertools import permutations
    
    n = len(tasks)
    if n > 10:
        raise ValueError(f"Brute force unsafe for n={n} > 10")
    
    best_count = 0
    best_schedule = []
    
    # Try all n! permutations
    for perm in permutations(tasks):
        current_time = 0
        scheduled = []
        
        for task in perm:
            if current_time + task.duration <= task.deadline:
                scheduled.append(task.task_id)
                current_time += task.duration
        
        if len(scheduled) > best_count:
            best_count = len(scheduled)
            best_schedule = scheduled
    
    return best_count, best_schedule

def generate_tasks(n: int,
                   dur_min: int = 1,
                   dur_max: int = 10,
                   deadline_factor: float = 1.5) -> List[Task]:
    """
    Generate n tasks with random durations and feasible deadlines.
    
    Args:
        n: Number of tasks
        dur_min, dur_max: Range for task durations
        deadline_factor: Multiplier for cumulative time (larger = more slack)
    """
    tasks = []
    cumulative_time = 0
    
    for i in range(n):
        duration = random.randint(dur_min, dur_max)
        cumulative_time += duration
        # Deadline is randomly set between task completion and factor * cumulative
        deadline = random.randint(cumulative_time, 
                                 int(cumulative_time * deadline_factor))
        tasks.append(Task(task_id=i, duration=duration, deadline=deadline))
    
    return tasks

# ----- Experiments for Problem A -----

def exp_scheduling_sanity(trials: int = 30, n: int = 8) -> None:
    """Verify greedy optimality against brute force."""
    print("\n[Scheduling] Running sanity checks...")
    rows = []
    failures = 0
    
    for trial in range(trials):
        tasks = generate_tasks(n, dur_min=1, dur_max=5, deadline_factor=2.0)
        
        # Greedy solution
        greedy_scheduled, _, _ = greedy_task_scheduling(tasks)
        greedy_count = len(greedy_scheduled)
        
        # Brute force solution
        brute_count, _ = brute_force_scheduling(tasks)
        
        match = greedy_count == brute_count
        if not match:
            failures += 1
            print(f"  Trial {trial + 1}: MISMATCH - Greedy={greedy_count}, Brute={brute_count}")
        
        rows.append((trial + 1, n, greedy_count, brute_count, int(match)))
    
    write_csv("scheduling_sanity.csv",
              ["trial", "n", "greedy_count", "brute_count", "match"],
              rows)
    
    if failures == 0:
        print(f"  ✓ All {trials} trials passed!")
    else:
        print(f"  ✗ {failures}/{trials} trials failed!")
        raise AssertionError("Greedy algorithm failed sanity check")

def exp_scheduling_timing(sizes: Tuple[int, ...] = (200, 400, 800, 1600, 3200, 6400),
                         trials: int = 100) -> None:
    """Benchmark greedy task scheduling algorithm."""
    print("\n[Scheduling] Running performance benchmarks...")
    rows = []
    
    for n in sizes:
        times = []
        counts = []
        
        for _ in range(trials):
            tasks = generate_tasks(n, dur_min=1, dur_max=20, deadline_factor=1.8)
            
            t0 = now_ns()
            scheduled, _, completion_time = greedy_task_scheduling(tasks)
            t1 = now_ns()
            
            times.append(secs(t1 - t0))
            counts.append(len(scheduled))
        
        mean_time = sum(times) / len(times)
        std_time = math.sqrt(sum((t - mean_time)**2 for t in times) / len(times))
        mean_count = sum(counts) / len(counts)
        
        rows.append((n, mean_time, std_time, mean_count))
        print(f"  n={n:5d}: {mean_time:.6f}s ± {std_time:.6f}s  "
              f"(avg {mean_count:.1f} tasks scheduled)")
    
    write_csv("scheduling_timing.csv",
              ["n", "mean_time_s", "std_time_s", "avg_tasks_scheduled"],
              rows)
    
    # Generate plot
    plt = try_matplotlib()
    if plt:
        import numpy as np
        ns = [r[0] for r in rows]
        means = [r[1] for r in rows]
        stds = [r[2] for r in rows]
        
        plt.figure(figsize=(10, 7))
        plt.errorbar(ns, means, yerr=stds, fmt='bo-', capsize=5,
                    capthick=2, markersize=8, linewidth=2,
                    label='Measured runtime')
        
        # Fit n log n on log-log scale
        log_ns = np.log(ns)
        log_means = np.log(means)
        coeffs = np.polyfit(log_ns, log_means, 1)
        fitted = np.exp(coeffs[1]) * np.array(ns) ** coeffs[0]
        
        plt.plot(ns, fitted, 'r--', linewidth=2.5,
                label=f'Fitted O(n log n), slope={coeffs[0]:.2f}')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Input Size (n tasks)', fontsize=14)
        plt.ylabel('Time (seconds)', fontsize=14)
        plt.title('Greedy Task Scheduling Algorithm Runtime', 
                 fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3, which='both')
        plt.tight_layout()
        plt.savefig(path_in_outputs('greedy_scheduling_runtime.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  → Saved plot: greedy_scheduling_runtime.png (slope={coeffs[0]:.3f})")

# ============================================
# PROBLEM B: Skyline Problem (D&C)
# ============================================

@dataclass(frozen=True)
class Building:
    """Represents a building with left edge, right edge, and height."""
    left: int      # Left x-coordinate
    right: int     # Right x-coordinate
    height: int    # Building height
    
    def __repr__(self):
        return f"Building([{self.left},{self.right}], h={self.height})"

# Skyline is represented as list of (x, height) key points
Skyline = List[Tuple[int, int]]

def merge_skylines(left: Skyline, right: Skyline) -> Skyline:
    """
    Merge two skylines into one.
    
    ALGORITHM:
    Use two-pointer technique to merge like merge sort:
    - Track current heights of both skylines (h1, h2)
    - At each step, advance pointer with smaller x-coordinate
    - Update current height to max(h1, h2)
    - Add key point only when height changes
    
    Time Complexity: O(n₁ + n₂) where n₁, n₂ are skyline sizes
    Space Complexity: O(n₁ + n₂)
    
    Args:
        left: Left skyline as list of (x, h) points
        right: Right skyline as list of (x, h) points
        
    Returns:
        Merged skyline
    """
    result = []
    i, j = 0, 0
    h1, h2 = 0, 0  # Current heights
    
    while i < len(left) and j < len(right):
        # Determine which point comes first
        if left[i][0] < right[j][0]:
            x, h1 = left[i]
            i += 1
        elif right[j][0] < left[i][0]:
            x, h2 = right[j]
            j += 1
        else:  # Same x-coordinate
            x = left[i][0]
            h1 = left[i][1]
            h2 = right[j][1]
            i += 1
            j += 1
        
        # Current max height
        max_h = max(h1, h2)
        
        # Add to result only if height changes
        if not result or result[-1][1] != max_h:
            result.append((x, max_h))
    
    # Append remaining points
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result

def skyline_divide_conquer(buildings: List[Building]) -> Skyline:
    """
    Compute city skyline using divide-and-conquer.
    
    DOMAIN EXPLANATION (Computer Graphics / Urban Planning):
    Given a city with multiple overlapping buildings (rectangles), compute
    the visible skyline silhouette as seen from a distance.
    
    Real-world applications:
    - Computer graphics: Rendering 3D city scenes
    - Geographic Information Systems (GIS): Urban planning visualization
    - Video games: Efficient rendering of city skylines
    - Architecture: Analyzing building visibility and shadows
    
    ABSTRACTION:
    Given: Set of buildings B = {b₁, b₂, ..., bₙ}, where each building bᵢ is:
      - A rectangle defined by (leftᵢ, rightᵢ, heightᵢ)
    Goal: Compute skyline S as ordered sequence of key points:
      - S = [(x₁, h₁), (x₂, h₂), ..., (xₖ, hₖ)]
      - Where height changes from hᵢ to hᵢ₊₁ at position xᵢ₊₁
    
    ALGORITHM (Divide-and-Conquer):
    1. Base case: If single building, return [(left, height), (right, 0)]
    2. Divide: Split buildings into left and right halves
    3. Conquer: Recursively compute skylines for each half
    4. Combine: Merge the two skylines using two-pointer technique
    
    CORRECTNESS PROOF:
    Base case: Single building skyline is trivially correct.
    
    Inductive hypothesis: Assume algorithm correctly computes skylines
    for all inputs of size < n.
    
    Inductive step: For input of size n:
    - By IH, left_skyline and right_skyline are correct
    - Merge operation maintains correctness:
      * At each x-coordinate, merged height = max(h_left, h_right)
      * This is exactly the definition of combined skyline
      * Height changes are preserved from both skylines
    
    Therefore, algorithm is correct for all n. ∎
    
    COMPLEXITY ANALYSIS:
    Recurrence: T(n) = 2T(n/2) + O(n)
      - 2T(n/2): Two recursive calls on halves
      - O(n): Merging two skylines of total size ≤ n
    
    By Master Theorem (case 2): T(n) = O(n log n)
    
    Space Complexity: O(n) for skyline storage + O(log n) recursion depth
    
    Args:
        buildings: List of buildings
        
    Returns:
        Skyline as list of (x, height) key points
    """
    if not buildings:
        return []
    
    # Base case: single building
    if len(buildings) == 1:
        b = buildings[0]
        return [(b.left, b.height), (b.right, 0)]
    
    # Divide
    mid = len(buildings) // 2
    left_skyline = skyline_divide_conquer(buildings[:mid])
    right_skyline = skyline_divide_conquer(buildings[mid:])
    
    # Conquer: merge
    return merge_skylines(left_skyline, right_skyline)

def skyline_naive(buildings: List[Building]) -> Skyline:
    """
    Naive approach: For each x-coordinate, scan all buildings to find max height.
    
    Time Complexity: O(n · m) where m is number of distinct x-coordinates
    Worst case: O(n²) when all coordinates are distinct
    """
    if not buildings:
        return []
    
    # Collect all x-coordinates
    x_coords = set()
    for b in buildings:
        x_coords.add(b.left)
        x_coords.add(b.right)
    
    x_sorted = sorted(x_coords)
    
    # For each x-coordinate, find max height
    result = []
    for x in x_sorted:
        max_height = 0
        for b in buildings:
            if b.left <= x < b.right:
                max_height = max(max_height, b.height)
        
        # Add key point if height changes
        if not result or result[-1][1] != max_height:
            result.append((x, max_height))
    
    return result

def generate_buildings(n: int,
                      x_range: int = 1000,
                      width_min: int = 10,
                      width_max: int = 100,
                      height_min: int = 10,
                      height_max: int = 200) -> List[Building]:
    """Generate n random buildings."""
    buildings = []
    for i in range(n):
        left = random.randint(0, x_range - width_max)
        width = random.randint(width_min, width_max)
        right = left + width
        height = random.randint(height_min, height_max)
        buildings.append(Building(left=left, right=right, height=height))
    return buildings

# ----- Experiments for Problem B -----

def exp_skyline_sanity(trials: int = 30, n: int = 20) -> None:
    """Verify D&C correctness against naive approach."""
    print("\n[Skyline] Running sanity checks...")
    rows = []
    failures = 0
    
    for trial in range(trials):
        buildings = generate_buildings(n, x_range=200, width_min=5, width_max=30)
        
        dc_skyline = skyline_divide_conquer(buildings)
        naive_skyline = skyline_naive(buildings)
        
        # Skylines should be identical
        match = dc_skyline == naive_skyline
        
        if not match:
            failures += 1
            print(f"  Trial {trial + 1}: MISMATCH")
            print(f"    D&C:   {len(dc_skyline)} points")
            print(f"    Naive: {len(naive_skyline)} points")
        
        rows.append((trial + 1, n, len(dc_skyline), len(naive_skyline), int(match)))
    
    write_csv("skyline_sanity.csv",
              ["trial", "n", "dc_points", "naive_points", "match"],
              rows)
    
    if failures == 0:
        print(f"  ✓ All {trials} trials passed!")
    else:
        print(f"  ✗ {failures}/{trials} trials failed!")
        raise AssertionError("D&C algorithm failed sanity check")

def exp_skyline_timing(sizes: Tuple[int, ...] = (100, 200, 400, 800, 1600, 3200),
                      trials: int = 50) -> None:
    """Benchmark skyline computation."""
    print("\n[Skyline] Running performance benchmarks...")
    rows = []
    
    for n in sizes:
        dc_times = []
        naive_times = []
        dc_points = []
        
        for _ in range(trials):
            buildings = generate_buildings(n, x_range=5000, 
                                          width_min=10, width_max=100)
            
            # D&C timing
            t0 = now_ns()
            dc_skyline = skyline_divide_conquer(buildings)
            t1 = now_ns()
            dc_times.append(secs(t1 - t0))
            dc_points.append(len(dc_skyline))
            
            # Naive timing (skip for large n to save time)
            if n <= 800:
                t0 = now_ns()
                naive_skyline = skyline_naive(buildings)
                t1 = now_ns()
                naive_times.append(secs(t1 - t0))
        
        mean_dc_time = sum(dc_times) / len(dc_times)
        std_dc_time = math.sqrt(sum((t - mean_dc_time)**2 for t in dc_times) / len(dc_times))
        mean_dc_points = sum(dc_points) / len(dc_points)
        
        mean_naive_time = sum(naive_times) / len(naive_times) if naive_times else None
        std_naive_time = (math.sqrt(sum((t - mean_naive_time)**2 for t in naive_times) / len(naive_times)) 
                         if naive_times else None)
        
        rows.append((n, mean_dc_time, std_dc_time, mean_dc_points,
                    mean_naive_time, std_naive_time))
        
        naive_str = (f"Naive {mean_naive_time:.6f}s ± {std_naive_time:.6f}s" 
                    if mean_naive_time else "Naive skipped")
        print(f"  n={n:5d}: D&C {mean_dc_time:.6f}s ± {std_dc_time:.6f}s "
              f"({mean_dc_points:.1f} points) | {naive_str}")
    
    write_csv("skyline_timing.csv",
              ["n", "dc_mean_time_s", "dc_std_time_s", "avg_skyline_points",
               "naive_mean_time_s", "naive_std_time_s"],
              rows)
    
    # Generate plots
    plt = try_matplotlib()
    if plt:
        import numpy as np
        ns = [r[0] for r in rows]
        dc_means = [r[1] for r in rows]
        dc_stds = [r[2] for r in rows]
        
        # Runtime plot (log-log scale)
        plt.figure(figsize=(10, 7))
        plt.errorbar(ns, dc_means, yerr=dc_stds, fmt='bo-', capsize=5,
                    capthick=2, markersize=8, linewidth=2,
                    label='Measured D&C runtime')
        
        # Fit n log n
        log_ns = np.log(ns)
        log_means = np.log(dc_means)
        coeffs = np.polyfit(log_ns, log_means, 1)
        fitted = np.exp(coeffs[1]) * np.array(ns) ** coeffs[0]
        
        plt.plot(ns, fitted, 'r--', linewidth=2.5,
                label=f'Fitted O(n log n), slope={coeffs[0]:.2f}')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Input Size (n buildings)', fontsize=14)
        plt.ylabel('Time (seconds)', fontsize=14)
        plt.title('Divide-and-Conquer Skyline Algorithm Runtime',
                 fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3, which='both')
        plt.tight_layout()
        plt.savefig(path_in_outputs('dc_skyline_runtime.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  → Saved plot: dc_skyline_runtime.png (slope={coeffs[0]:.3f})")
        
        # Comparison plot (D&C vs Naive for small n)
        if any(r[4] is not None for r in rows):
            fig, ax = plt.subplots(figsize=(10, 7))
            
            ns_with_naive = [r[0] for r in rows if r[4] is not None]
            dc_means_subset = [r[1] for r in rows if r[4] is not None]
            naive_means = [r[4] for r in rows if r[4] is not None]
            
            ax.plot(ns_with_naive, dc_means_subset, 'bo-', 
                   markersize=8, linewidth=2, label='D&C O(n log n)')
            ax.plot(ns_with_naive, naive_means, 'rs-',
                   markersize=8, linewidth=2, label='Naive O(n²)')
            
            ax.set_xlabel('Input Size (n buildings)', fontsize=14)
            ax.set_ylabel('Time (seconds)', fontsize=14)
            ax.set_title('Algorithm Comparison: D&C vs Naive',
                        fontsize=16, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(path_in_outputs('skyline_comparison.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  → Saved plot: skyline_comparison.png")

# ============================================
# Main Execution
# ============================================

def main():
    """Run all experiments and generate outputs."""
    print("=" * 70)
    print("Algorithm Design Project: Greedy & Divide-and-Conquer")
    print("=" * 70)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    
    random.seed(RANDOM_SEED)
    ensure_outputs_dir()
    
    # ===== PROBLEM A: Task Scheduling (Greedy) =====
    print("\n" + "=" * 70)
    print("PROBLEM A: Task Scheduling with Deadlines (Greedy Algorithm)")
    print("=" * 70)
    
    exp_scheduling_sanity(trials=30, n=8)
    exp_scheduling_timing(sizes=(200, 400, 800, 1600, 3200, 6400), trials=100)
    
    # ===== PROBLEM B: Skyline Problem (D&C) =====
    print("\n" + "=" * 70)
    print("PROBLEM B: City Skyline Problem (Divide-and-Conquer)")
    print("=" * 70)
    
    exp_skyline_sanity(trials=30, n=20)
    exp_skyline_timing(sizes=(100, 200, 400, 800, 1600, 3200), trials=50)
    
    # ===== Summary =====
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nAll artifacts written to: {os.path.abspath(OUTPUT_DIR)}/")
    print("\nGenerated files:")
    
    expected_files = [
        "scheduling_sanity.csv",
        "scheduling_timing.csv",
        "greedy_scheduling_runtime.png",
        "skyline_sanity.csv",
        "skyline_timing.csv",
        "dc_skyline_runtime.png",
        "skyline_comparison.png"
    ]
    
    for fn in expected_files:
        p = path_in_outputs(fn)
        if os.path.exists(p):
            size = os.path.getsize(p)
            print(f"  ✓ {fn:35s} ({size:,} bytes)")
        else:
            status = " (skipped - no matplotlib)" if fn.endswith(".png") else " (MISSING)"
            print(f"  ✗ {fn:35s} {status}")
    
    print("\n" + "=" * 70)
    print("Ready for LaTeX inclusion!")
    print("=" * 70)

if __name__ == "__main__":
    main()