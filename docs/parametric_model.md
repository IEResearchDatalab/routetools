# Reverse-Engineered Parametric Performance Model

This document describes the closed-form parametric model reverse-engineered
from the SWOPP3 performance model binary (`swopp3_performance_model`).

## Overview

The SWOPP3 model exposes two scalar functions:

| Function | Description |
|----------|-------------|
| `predict_no_wps(tws, twa, swh, mwa, v)` | Power without Wind Propulsion System |
| `predict_with_wps(tws, twa, swh, mwa, v)` | Power with Wind Propulsion System (sails) |

Both return propulsion power in **kW** and accept:

| Parameter | Symbol | Unit | Range |
|-----------|--------|------|-------|
| True wind speed | $\text{tws}$ | m/s | $[0, 30]$ |
| True wind angle | $\text{twa}$ | deg | $[0, 180]$ (symmetric) |
| Significant wave height | $\text{swh}$ | m | $[0, 10]$ |
| Mean wave angle | $\text{mwa}$ | deg | $[0, 180]$ (symmetric) |
| Ship speed | $v$ | m/s | $[0, 14.5]$ |

## `predict_no_wps` — Closed-Form Model

The total power is the sum of three **perfectly additive** components,
clamped at zero:

$$
P_{\text{no\_wps}} = \max\!\Big(0,\; P_{\text{hull}} + P_{\text{wind}} + P_{\text{wave}}\Big)
$$

### Hull Resistance

Pure cubic dependence on ship speed, independent of environment:

$$
P_{\text{hull}} = K_h \cdot v^3
$$

$$
K_h = \frac{969}{226} \approx 4.28761
$$

**Physical interpretation:** Hydrodynamic drag in calm water. The cubic law
follows from drag force $\propto v^2$ multiplied by speed to get power.

**Literature validation:** The cubic dependence is consistent with the following formula found in naval papers:

$$
P_{\text{base}} = \frac{\Delta^{2/3} \cdot v^3}{3.7 \left( \sqrt{L} + 75 / v \right) }
$$

where $\Delta$ is displacement and $L$ is the length of the vessel.

**References:**

- Molland, A.F., Turnock, S.R., Hudson, D.A. (2017). *Ship Resistance and Propulsion*, 2nd ed. Cambridge University Press. Chapter 3. [doi:10.1017/9781316494196](https://doi.org/10.1017/9781316494196)
- Holtrop, J., Mennen, G.G.J. (1982). "An approximate power prediction method." *International Shipbuilding Progress*, 29(335), 166–170. [doi:10.3233/ISP-1982-2933501](https://doi.org/10.3233/ISP-1982-2933501)

### Aerodynamic (Wind) Resistance

Depends on the **apparent wind** seen by the ship:

$$
P_{\text{wind}} = K_a \cdot v \cdot \big(V_R \cdot u_x - v^2\big)
$$

where the apparent wind components are:

$$
u_x = \text{tws} \cdot \cos\!\left(\text{twa} \cdot \frac{\pi}{180}\right) + v
\qquad
u_y = \text{tws} \cdot \sin\!\left(\text{twa} \cdot \frac{\pi}{180}\right)
$$

$$
V_R = \sqrt{u_x^2 + u_y^2}
$$

$$
K_a = \frac{49}{320} = 0.153125
$$

**Physical interpretation:**
$K_a = \frac{1}{2} \cdot \rho_{\text{air}} \cdot C_D A \cdot \frac{1}{1000}$
where $\rho_{\text{air}} = 1.225 \;\text{kg/m}^3$ and $C_D A = 250 \;\text{m}^2$.
The factor $1/1000$ converts W to kW.

**Note:** At large TWA with strong tailwinds, $P_{\text{wind}}$ becomes
negative (wind assists the ship), which can drive total power to zero (clamped).

**Literature validation:** The formulation is consistent with the standard aerodynamic wind load model used in naval architecture:

$$
P_{\text{wind}} =
\frac{1}{2} \cdot C_X \cdot \rho_{\text{air}} \cdot A_x \cdot
\cos(\phi) \cdot v_{\text{wind}}^2 \cdot v
$$

which follows from the classical drag expression

$$
F_{\text{wind}} =
\frac{1}{2} \, \rho_{\text{air}} \, C_X \, A_x \, v_{\text{wind}}^2
$$

with power obtained as $P = F \cdot v$.
Here $C_X$ is the longitudinal aerodynamic force coefficient, $A_x$ is the projected frontal area, and $\phi$ accounts for the wind attack angle.

**References:**

- Blendermann, W. (1994). "Parameter identification of wind loads on ships." *Journal of Wind Engineering and Industrial Aerodynamics*, 51(3), 339–351. [doi:10.1016/0167-6105(94)90067-1](https://doi.org/10.1016/0167-6105(94)90067-1)
- Fujiwara, T., Ueno, M., Nimura, T. (1998). "Estimation of wind forces and moments acting on ships." *Journal of the Society of Naval Architects of Japan*, 183, 77–90. [doi:10.2534/jjasnaoe1968.1998.77](https://doi.org/10.2534/jjasnaoe1968.1998.77)
- ITTC (2014). "Recommended Procedures and Guidelines: Speed and Power Trials." 7.5-04-01-01.1. [ittc.info](https://www.ittc.info/media/8370/75-04-01-011.pdf)

### Wave-Added Resistance

Factorizes cleanly into three independent terms:

$$
P_{\text{wave}} = A_w \cdot \text{swh}^2 \cdot v^{3/2} \cdot \exp\!\Big(-k_w \cdot |\theta_{\text{mwa}}|^3\Big)
$$

where $\theta_{\text{mwa}} = \text{mwa} \cdot \pi / 180$ is the wave angle in radians.

$$
A_w \approx 11.1395 \qquad k_w \approx 0.28935
$$

**Key properties:**

- Quadratic in SWH ($\propto \text{swh}^2$) — exact
- Speed exponent is exactly $3/2$ ($\propto v^{1.5}$)
- Directional factor $\exp(-k_w |\theta|^3)$ decays from 1.0 at head seas
  ($\text{mwa}=0°$) to $\approx 0.00013$ at following seas ($\text{mwa}=180°$)

**Literature validation:** The structure is consistent with semi-empirical added resistance formulations derived from strip theory and experiments, where added resistance scales approximately with wave height squared and increases with ship speed:

$$
R_{\text{aw}} \propto \rho_{\text{water}} \cdot g \cdot H^2 \cdot f(v, \beta)
$$

and the associated power is

$$
P_{\text{wave}} = R_{\text{aw}} \cdot v
$$

Quadratic dependence on wave height follows from linear wave theory, since wave energy is proportional to $H^2$. Experiments typically report that added resistance grows with speed somewhere between linearly ($\propto v$) and quadratically ($\propto v^2$), which makes the fitted exponent of $v^{3/2}$ physically plausible. Strong directional decay from head to following seas is also consistent with measured added resistance characteristics.

**References:**

- Gerritsma, J., Beukelman, W. (1972). "Analysis of the resistance increase in waves of a fast cargo ship." *International Shipbuilding Progress*, 19(217), 285–293. [doi:10.3233/ISP-1972-1921701](https://journals.sagepub.com/doi/abs/10.3233/ISP-1972-1921701)
- Salvesen, N. (1978). "Added resistance of ships in waves." *Journal of Hydronautics*, 12(1), 24–34. [doi:10.2514/3.63110](https://doi.org/10.2514/3.63110)
- Faltinsen, O.M., Minsaas, K.J., Liapis, N., Skjørdal, S.O. (1980). "Prediction of resistance and propulsion of a ship in a seaway." *Proc. 13th Symposium on Naval Hydrodynamics*, Tokyo, 505–529. ([Google Scholar](https://scholar.google.com/scholar?q=%22Prediction+of+resistance+and+propulsion+of+a+ship+in+a+seaway%22+Faltinsen+1980) — conference proceedings, no DOI available)

### Accuracy

Tested against the reference binary on 10,000 random inputs:

| Metric | Value |
|--------|-------|
| Mean absolute error | 0.008 kW |
| p99 absolute error | 0.064 kW |
| Max absolute error | 0.114 kW |
| Max relative error | 0.031% |

---

## `predict_with_wps` — Sail-Assisted Model (Closed-Form)

$$
P_{\text{with\_wps}} = \max\!\Big(0,\; P_{\text{hull}} + P_{\text{wind}} + P_{\text{wave}} - P_{\text{sail}}\Big)
$$

where hull, wind, and wave terms are identical to `predict_no_wps`.

### Sail Power — Closed-Form

The sail power saving factorizes exactly as:

$$
P_{\text{sail}} = C(\text{AWA}) \cdot V_R^2 \cdot v
$$

where $V_R^2 = u_x^2 + u_y^2$ is the squared apparent wind speed
and $\text{AWA}$ is the apparent wind angle:

$$
\text{AWA} = \arctan2\!\big(|u_y|,\; u_x\big) \cdot \frac{180}{\pi}
$$

so $\text{AWA}$ is in **degrees**.

**Sail polar coefficient** $C(\text{AWA})$:

$$
C(\text{AWA}) = \begin{cases}
0 & \text{if } \text{AWA} < 10° \\[4pt]
K_s \cdot \sin\alpha \cdot \Big(1 + \dfrac{3}{20}\sin^2\alpha\Big) & \text{if } \text{AWA} \geq 10°
\end{cases}
$$

where $\alpha = (\text{AWA} - 10°) \cdot \pi / 180$ (converted to radians) and:

$$
K_s = 0.85903125
$$

### Key Properties

1. **Wave-independent:** $P_{\text{sail}}$ depends only on
   $(\text{tws}, \text{twa}, v)$ — no dependence on SWH or MWA.

2. **Operates in apparent wind coordinates:** The model naturally uses
   $\text{AWA}$ and $V_R$, not the true wind quantities directly.

3. **10° dead zone:** The sail produces no power when $\text{AWA} < 10°$
   (too close to head wind). This cutoff is exact in apparent wind angle.

4. **Peak at AWA ≈ 100°:** $C(100°) = K_s \cdot 23/20 \approx 0.9879$ —
   the peak of the sail thrust polar occurs at beam-reach conditions.

5. **Always non-negative:** $P_{\text{sail}} \geq 0$ (sails only help).

6. **Symmetric:** $P_{\text{sail}}(\text{twa}) = P_{\text{sail}}(-\text{twa})$.

7. **Physical interpretation:** $K_s \cdot \sin\alpha$ is the primary
   lift-based thrust; the $\frac{3}{20}\sin^2\alpha$ term is a quadratic
   drag/lift correction that enhances thrust at large angles of attack.

### Accuracy

The closed-form sail model is **exact to machine precision** ($< 10^{-13}$)
against the reference. The only residual error in the full model comes from
the $A_w$ and $k_w$ constants in the wave term.

Tested against the reference binary on 50,000 random inputs:

| Metric | Value |
|--------|-------|
| Mean absolute error | 0.004 kW |
| p99 absolute error | 0.030 kW |
| Max absolute error | 0.050 kW |
| Errors > 0.1 kW | 0 / 50,000 |

---

## Summary of Constants

| Constant | Symbol | Value | Exact |
|----------|--------|-------|-------|
| Hull coefficient | $K_h$ | 4.28761… | $969/226$ |
| Air drag coefficient | $K_a$ | 0.153125 | $49/320$ |
| Wave amplitude | $A_w$ | 11.1395 | fitted |
| Wave directional decay | $k_w$ | 0.28935 | fitted |
| Sail thrust coefficient | $K_s$ | 0.85903125 | exact |
| Sail dead zone angle | — | 10° | exact |
| Sail quadratic correction | — | 3/20 = 0.15 | exact |

## Clamping Rule

Both functions clamp the result at zero:

$$
P = \max(0, P_{\text{raw}})
$$

This occurs when strong tailwinds ($P_{\text{wind}} < 0$) or large sail
savings ($P_{\text{sail}}$) exceed hull + wave resistance.
