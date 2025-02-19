# UniquenessGraphicalMethod

This repository contains the relevant code and resources related to the proof that the optimal control is unique and independent of the convex cost function. The proof is structured using a set of lemmas that establish key properties of the optimal generated energy function 𝑊(𝑡).

## Contents:
* Proof Diagram: The proof_diagram.pdf file provides a visual representation of the proof structure.
* Code Implementation: Relevant scripts to illustrate the theoretical results numerically.
* Mathematical Lemmas:
  * Lemma 1 (Straight Flow): The optimal generated power remains constant unless constrained by the energy requirement.
  * Lemma 2 (Tangent & Continuous): The curve of optimal generated energy must be tangent to its bounding constraints.
  * Lemma 3 (Up-Increasing, Down-Decreasing): The optimal energy cannot remain at the upper bound when the power is decreasing, nor at the lower bound when the power is increasing.
  * Lemma 4 (Uniqueness Between Monotones): If the optimal energy moves in a straight line between two constrained intervals, this segment is unique.
  * Lemma 5 (Most Distant Reachable Interval): The optimal energy must move in a straight line to the most distant monotone interval possible.
  * Final Proof Statement: Establishes that the optimal generated energy function is unique due to its behavior in constrained and unconstrained intervals.
