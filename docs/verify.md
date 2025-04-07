## 3. LP-based Algorithm for m-Height Problem

In this section, we present an algorithm that solves the "m-height problem" based on linear programming (LP). The proof for the algorithm is provided in reference [Jia24].

Define:
- Let \([n⟩ = \{0, 1, \dots, n−1\}\) for any positive integer \(n\).
- Let \(\Psi = \{-1, 1\}^m\) be the set of \(2^m\) binary vectors of length \(m\) whose elements are either 1 or -1.

Consider the tuple \((a, b, X, \psi)\) where:
- \(a \in [n⟩\)
- \(b \in [n⟩ \setminus \{a\}\)
- \(X \subseteq [n⟩ \setminus \{a, b\}\)
- \(|X| = m−1\)
- \(\psi = (s_0, s_1, \dots, s_{m-1}) \in \Psi\)

Let \(\Gamma\) denote the set of all such tuples, with size \(n(n−1)\binom{n−2}{m−1}2^m\).

Given \((a, b, X, \psi)\in\Gamma\), define \(Y \equiv [n⟩ \setminus X \setminus \{a, b\}\), and let \(\tau\) be the permutation on \([n⟩\) such that \(\tau(j) = x_j\), where \(x_0 = a\), \(x_m = b\), and the integers in \(X\) and \(Y\) are sorted increasingly.

### Theorem 1 (LP-based Solution):
Let \(m \in \{1, 2, \dots, \min\{d(C), n−1\}\}\). Given \((a, b, X, \psi) \in \Gamma\), define the linear program \(LP_{a,b,X,\psi}\) as follows:

#### Variables:
- Real-valued variables \(u_0, u_1, \dots, u_{k−1}\).

#### Objective:
\[
\text{maximize } \sum_{i \in [k⟩}(s_0 g_{i,a}) \cdot u_i
\]

#### Constraints:
\[
\sum_{i \in [k⟩}(s_{\tau^{-1}(j)} g_{i,j} − s_0 g_{i,a}) \cdot u_i \leq 0 \quad \text{for } j \in X
\]
\[
\sum_{i \in [k⟩}(−s_{\tau^{-1}(j)} g_{i,j}) \cdot u_i \leq −1 \quad \text{for } j \in X
\]
\[
\sum_{i \in [k⟩} g_{i,b} \cdot u_i = 1
\]
\[
\sum_{i \in [k⟩} g_{i,j} \cdot u_i \leq 1 \quad \text{for } j \in Y
\]
\[
\sum_{i \in [k⟩} −g_{i,j} \cdot u_i \leq 1 \quad \text{for } j \in Y
\]

Let \(z_{a,b,X,\psi}\) denote the optimal objective value of \(LP_{a,b,X,\psi}\), set \(z_{a,b,X,\psi} = \infty\) if unbounded, and \(z_{a,b,X,\psi} = 0\) if infeasible. Then the \(m\)-height of code \(C\) is:
\[
h_m(C) = \max_{(a,b,X,\psi) \in \Gamma} z_{a,b,X,\psi}
\]

### Algorithm to Compute \(h_m(C)\):

1. Set \(h_0(C) = 1\).
2. For \(m = 1, 2, 3, \dots\), compute:
\[
h_m(C) = \max_{(a,b,X,\psi) \in \Gamma} z_{a,b,X,\psi}
\]
Stop as soon as \(h_{m^*}(C) = \infty\) is encountered. Set \(d(C) = m^*\).
3. For \(m > m^*\), set \(h_m(C) = \infty\).

