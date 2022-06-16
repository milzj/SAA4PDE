---
date: April 15, 2022
title: Termination criteria
---

We relate the "termination criterion"
$$\\|\alpha v + F(\mathrm{prox}\_{\varphi/\alpha}(v))\\|\_{U} \leq \varepsilon$$
to
$$\\|u-\mathrm{prox}\_{\varphi/\alpha}(-(1/\alpha)F(u))\\|\_{U} \leq \delta,$$
where $u = \mathrm{prox}\_{\varphi/\alpha}(v)$. More precisely, given the
validity of the first estimate, we derive an upper bound on $\delta$
such that the second inequality holds true.

The evaluation of the first inequality requires the point $v$. The
second one can be evaluated using
$u = \mathrm{prox}\_{\varphi/\alpha}(v)$. The mapping
$v \mapsto \alpha v+F(\mathrm{prox}\_{\varphi/\alpha}(v))$ defines a
normal map.

Let $U$ be a real Hilbert space and let $v \in U$. Moreover let
$\varphi : U \to (-\infty,\infty]$ be proper, closed, and convex, let
$\alpha > 0$, and let $F : U \to U$ be a mapping. We define
$u = \mathrm{prox}\_{\varphi/\alpha}(v)$. Here
$\mathrm{prox}\_{\varphi/\alpha}(\cdot)$ is the proximity operator of the
function $\varphi/\alpha$.

Suppose that
$\\|\alpha v + F(\mathrm{prox}\_{\varphi/\alpha}(v))\\|\_{U} \leq \varepsilon$
for some $\varepsilon \geq 0$. Since
$\mathrm{prox}\_{\varphi/\alpha}(\cdot)$ is firmly nonexpansive, we have

$$\begin{aligned}
    \\|u-\mathrm{prox}\_{\varphi/\alpha}(-(1/\alpha)F(u))\\|\_{U}
    & = 
    \\|\mathrm{prox}\_{\varphi/\alpha}(v)-\mathrm{prox}\_{\varphi/\alpha}(-(1/\alpha)F(u))\\|\_{U}
    \\
    & \leq
    \\|v+(1/\alpha)F(u)\\|\_{U}
    \\
    & = (1/\alpha)\\|\alpha v+ F(u)\\|\_{U}
    \\
    & = (1/\alpha)\\|\alpha v+ F(\mathrm{prox}\_{\varphi/\alpha}(v))\\|\_{U}
    \\
    & \leq \varepsilon/\alpha.\end{aligned}$$

If $\varphi = 0$, then the first estimate is an identity.
