= Report, Project 1

Name: Yuuzen Shen
UIN: 434000618

== Algorithm

Our algorithm has three parts:

1. The generator that is piecewise for generating the matrix from brute force (for m<=3) and simulated annealing (for m>3)
2. The checker that checks the validity of the generated matrix based on given LP solution
3. A annealing generator that uses simulated annealing to generate the matrix from a seed

The part 1 provided preliminary seeds for the annealing generator.