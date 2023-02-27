
For each $x \in [0,1]^2$, the expectation of the random diffusion coefficient

$$
\kappa(\xi)(x) = 
\exp\big(\xi_1 \cos(1.1\pi x_1)
+\xi_2 \cos(1.2\pi x_1)
+\xi_3 \sin(1.3\pi x_2)
+\xi_4 \sin(1.4\pi x_2)\big)
$$

can be computed explicity, since $\xi_1, \ldots, \xi_4$ are independent
$[-1,1]$ uniform random variables. 

The expectation of the random diffusion coefficient is implemented in
[.avg_mknrandom_field.py](vg_mknrandom_field.py).


The expected value can be computed with the help of
[WolframAlpha](https://www.wolframalpha.com/input?i=integrate+exp%28+x+sin%28a%29%29%2F2+for+x%3D-1..1)
and
[WolframAlpha](https://www.wolframalpha.com/input?i=integrate+exp%28+x+cos%28a%29%29%2F2+for+x%3D-1..1).


