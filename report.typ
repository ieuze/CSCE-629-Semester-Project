= Report, Project 1

Name: Yuuzen Shen
UIN: 434000618

== Algorithm

Our algorithm has three parts:

1. The generator that is piecewise for generating the matrix from brute force (for m<=3) and simulated annealing (for m>3)
2. The checker that checks the validity of the generated matrix based on given LP solution
3. A annealing generator that uses simulated annealing to generate the matrix from a seed

The part 1 provided preliminary seeds for the annealing generator.

== Bound, and Complexity (if any)

Brute Force: Exact (within element range), Exponential time.
Annealing (Random Start): Heuristic (no guarantee), Time depends on annealing schedule and parallel runs, but does converge depends on ratio of temperature drops.
Random Search: Heuristic (no guarantee, weakest), Time depends on number of samples. Hard set by parameter.
Annealing (Seed Start): Heuristic (no guarantee, depends on seed), Time depends on annealing schedule and parallel neighbor evaluation. Returns second-best result as well for measure or reseeding.

== Main Idea

For a small m, we can brute force the solution by checking all possible values of the matrix.
For a larger m, we can use simulated annealing to generate a good seed, and then use the annealing generator to generate a good matrix.

Why not one run? Because this problem shows local maximum and global maximum, even though the simulated annealing is a heuristic made to combat that, it would be much faster to parallelize the runs via generating high quality seeds and annealing from them, rather than parallelizing the search, because the entropy introduced by the random seed is much more effective than the entropy introduced by the random search (if any).

== Very brief pseudocode




```typ
Input: k, n, m, element_range
Output: Best P matrix found

if m <= 3: // Threshold for 'small' m
  // Use Brute Force Generator
  P_best = BruteForceSearch(k, n, m, element_range)
else:
  // Manual Seed Generation Phase (Conceptual)
  // 1. Run Annealing Generator multiple times
  seeds_annealing = []
  for run in 1 to NUM_ANNEALING_RUNS:
    P_candidate = AnnealingGenerator(k, n, m, element_range, annealing_params)
    seeds_annealing.append(P_candidate)

  // 2. Run Random Generator multiple times
  seeds_random = []
  for run in 1 to NUM_RANDOM_RUNS:
    P_candidate = RandomGenerator(k, n, m, element_range, num_samples)
    seeds_random.append(P_candidate)

  // 3. Manually inspect/evaluate seeds_annealing and seeds_random
  //    Select the most promising seed based on m-height or other criteria.
  P_seed = HandpickBestSeed(seeds_annealing + seeds_random)

  // Seeded Annealing Phase
  // Write P_seed to a file (e.g., "seed.txt") along with n, k, m
  SaveSeedToFile("seed.txt", n, k, m, P_seed)

  // Run Seeded Annealing Generator using the chosen seed file
  P_best = SeededAnnealingGenerator(seed_file="seed.txt", annealing_params)

return P_best
```