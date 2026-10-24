"""Symbolic check of P1' (time-unit identifiability in the age-structured kinetic model).

Model: p(a | T) ∝ rho(T - a) * G(a)   (entry flux rho over entry time E = T - a, survival/growth G over age)
Emission depends on age only through mu(a) = closed-form kinetics; rescaling a = c*a', rates*c, maps the
emission family onto itself, so observational equivalence <=> the age laws match after rescaling:
    rho'(T - a') G'(a')  ∝_T  rho(T - c a') G(c a')          for all T, a'      (*)
Checks:
 1. d/dT d/da' of log(*) gives  (log rho')''(T - a') = c (log rho)''(T - c a').
 2. If log rho is quadratic (second derivative k), an explicit (rho', G') satisfies (*) for any c  -> not identified.
 3. If log rho is NOT quadratic (cubic term), the cross-derivative condition fails for c != 1.
 4. Compensating G' for the quadratic case contains a c(c-1)k a'^2/2 term: piecewise-constant growth
    rates (log G piecewise linear) cannot absorb it unless k = 0.
 5. Onset edge: rho = 0 for E < T0 -> support of a at time T is [0, T - T0]; under rescaling the support of a'
    is [0, (T - T0)/c], which must equal [0, T - T0'] for all T -> only c = 1.
 6. Emission: d mu/da = alpha - gamma mu; with a = c a' the curve mu(c a') solves the same ODE in a' with (c alpha, c gamma).
"""
import sympy as sp

T, a, c, k, b, b2, k3, u, T0, T0p = sp.symbols("T a c k b b2 k3 u T0 T0p", real=True)
alpha, gamma, mu0 = sp.symbols("alpha gamma mu0", positive=True)
logrho = sp.Function("logrho")
logrhop = sp.Function("logrhop")
logG = sp.Function("logG")
logGp = sp.Function("logGp")
kT = sp.Function("kT")

# 1. cross derivative
lhs = logrhop(T - a) + logGp(a)
rhs = logrho(T - c * a) + logG(c * a) + kT(T)
cross = sp.simplify(sp.diff(lhs - rhs, T, a))
print("1. cross-derivative of (*):", cross)

# 2. quadratic log rho: explicit equivalent pair for arbitrary c
lr = lambda x: k * x**2 / 2 + b * x
lrp = lambda x: c * k * x**2 / 2 + b2 * x
g0 = sp.Function("g0")
lGp = lambda x: g0(c * x) + k * c * (c - 1) * x**2 / 2 + (b2 - b * c) * x
resid = sp.expand(lrp(T - a) + lGp(a) - (lr(T - c * a) + g0(c * a)))
print("2. residual after compensation:", sp.factor(resid), "| d/da =", sp.simplify(sp.diff(resid, a)))

# 3. non-quadratic log rho: cubic term breaks the cross-derivative condition for c != 1
lr3 = k3 * u**3 / 6 + k * u**2 / 2 + b * u
d2 = sp.diff(lr3, u, 2)
cond = d2.subs(u, T - c * a) * c
v = sp.symbols("v", real=True)
dep = sp.simplify(sp.diff(cond.subs(T, v + a), a))
print("3. d/da of c*(log rho)''(T-ca) at fixed T-a (must be 0 for equivalence):", sp.factor(dep))

# 4. growth form needed in the quadratic case
print("4. compensating term in log G'(a'):", sp.factor(k * c * (c - 1) * a**2 / 2))

# 5. onset edge
coeffs = sp.Poly(sp.expand((T - T0) - c * (T - T0p)), T).coeffs()
print("5. support match (T-T0) = c (T-T0') for all T ->", sp.solve(coeffs, [c, T0p], dict=True))

# 6. emission closure under rescaling
mu = sp.Function("mu")
mu_a = sp.dsolve(sp.Eq(mu(a).diff(a), alpha - gamma * mu(a)), ics={mu(0): mu0}).rhs
ap = sp.symbols("ap", real=True)
mu_res = mu_a.subs(a, c * ap)
print("6. mu(c a') solves the ODE with rates (c alpha, c gamma):",
      sp.simplify(sp.diff(mu_res, ap) - (c * alpha - c * gamma * mu_res)) == 0)
