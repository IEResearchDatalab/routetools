# Reverse-Engineered Parametric Performance Model

This document describes the closed-form parametric model reverse-engineered
from the SWOPP3 performance model binary (`swopp3_performance_model`).

## Reference Vessel

The SWOPP3 performance model is calibrated for a specific vessel:

| Property                       | Value                           |
| ------------------------------ | ------------------------------- |
| Type                           | Single-skeg general cargo ship  |
| Length overall ($L$)           | 88 m                            |
| Estimated beam ($B$)           | ~15 m                           |
| Estimated wetted surface ($S$) | ~2200 m²                        |
| Propulsion                     | CPP, electric                   |
| Wingsails                      | 4 × 138 m² rigid (552 m² total) |

All coefficients below are derived from these specifications.

## Overview

The SWOPP3 model exposes two scalar functions:

| Function                                  | Description                               |
| ----------------------------------------- | ----------------------------------------- |
| `predict_no_wps(tws, twa, swh, mwa, v)`   | Power without Wind Propulsion System      |
| `predict_with_wps(tws, twa, swh, mwa, v)` | Power with Wind Propulsion System (sails) |

Both return propulsion power in **kW** and accept:

| Parameter               | Symbol       | Unit | Range                  |
| ----------------------- | ------------ | ---- | ---------------------- |
| True wind speed         | $\text{tws}$ | m/s  | $[0, 30]$              |
| True wind angle         | $\text{twa}$ | deg  | $[0, 180]$ (symmetric) |
| Significant wave height | $\text{swh}$ | m    | $[0, 10]$              |
| Mean wave angle         | $\text{mwa}$ | deg  | $[0, 180]$ (symmetric) |
| Ship speed              | $v$          | m/s  | $[0, 14.5]$            |

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

$$
K_h = \frac{1}{2} \cdot \rho_{\text{water}} \cdot S \cdot C_T \cdot \frac{1}{1000}
$$

With $\rho_{\text{water}} = 1025 \;\text{kg/m}^3$ and $S \approx 2200 \;\text{m}^2$:

$$
C_T = \frac{K_h}{\frac{1}{2} \cdot 1025 \cdot 2200 \cdot 10^{-3}} \approx 0.0038
$$

This total resistance coefficient is within the typical range $0.002\text{--}0.005$
for cargo ships at service speed.

**Literature validation:** The cubic dependence is consistent with the following formula found in naval papers:

$$
P_{\text{base}} = \frac{\Delta^{2/3} \cdot v^3}{3.7 \left( \sqrt{L} + 75 / v \right) }
$$

where $\Delta$ is displacement and $L$ is the length of the vessel.

**References:**

- Molland, A.F., Turnock, S.R., Hudson, D.A. (2017). _Ship Resistance and Propulsion_, 2nd ed. Cambridge University Press. Chapter 3. [doi:10.1017/9781316494196](https://doi.org/10.1017/9781316494196)
- Holtrop, J., Mennen, G.G.J. (1982). "An approximate power prediction method." _International Shipbuilding Progress_, 29(335), 166–170. [doi:10.3233/ISP-1982-2933501](https://doi.org/10.3233/ISP-1982-2933501)

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

$$
K_a = \frac{1}{2} \cdot \rho_{\text{air}} \cdot C_D \cdot A_T \cdot \frac{1}{1000}
$$

where $\rho_{\text{air}} = 1.225 \;\text{kg/m}^3$. The factor $1/1000$ converts W to kW.
Solving for the drag-area product:

$$
C_D \cdot A_T = \frac{K_a}{\frac{1}{2} \cdot 1.225 \cdot 10^{-3}} = 250 \;\text{m}^2
$$

With $C_D \approx 0.7$ (typical for cargo ship superstructure), the implied
frontal area is $A_T \approx 357 \;\text{m}^2$. For an 88 m vessel with
$\sim 15 \;\text{m}$ beam and $\sim 20 \;\text{m}$ air draught, this is
physically consistent ($15 \times 20 = 300 \;\text{m}^2$ plus rigging and
wingsail structure).

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

- Blendermann, W. (1994). "Parameter identification of wind loads on ships." _Journal of Wind Engineering and Industrial Aerodynamics_, 51(3), 339–351. [doi:10.1016/0167-6105(94)90067-1](<https://doi.org/10.1016/0167-6105(94)90067-1>)
- Fujiwara, T., Ueno, M., Nimura, T. (1998). "Estimation of wind forces and moments acting on ships." _Journal of the Society of Naval Architects of Japan_, 183, 77–90. [doi:10.2534/jjasnaoe1968.1998.77](https://doi.org/10.2534/jjasnaoe1968.1998.77)
- ITTC (2014). "Recommended Procedures and Guidelines: Speed and Power Trials." 7.5-04-01-01.1. [ittc.info](https://www.ittc.info/media/8370/75-04-01-011.pdf)

### Wave-Added Resistance

Factorizes cleanly into three independent terms:

$$
P_{\text{wave}} = A_w \cdot \text{swh}^2 \cdot v^{3/2} \cdot \exp\!\Big(-K_w \cdot |\theta_{\text{mwa}}|^3\Big)
$$

where $\theta_{\text{mwa}} = \text{mwa} \cdot \pi / 180$ is the wave angle in radians.

$$
A_w \approx 11.1395 \qquad K_w = \frac{125}{432} = \frac{5^3}{2^4 \cdot 3^3} \approx 0.28935
$$

**Key properties:**

- Quadratic in SWH ($\propto \text{swh}^2$) — exact
- Speed exponent is exactly $3/2$ ($\propto v^{1.5}$)
- Directional factor $\exp(-K_w |\theta|^3)$ decays from 1.0 at head seas
  ($\text{mwa}=0°$) to $\approx 0.00013$ at following seas ($\text{mwa}=180°$)

**Literature validation:** The three standard semi-empirical frameworks for
added resistance in waves all share the same dimensional structure:

- **Gerritsma & Beukelman (1972)** derive added resistance from strip theory
  as an integral of relative wave-induced motions along the hull. The result
  scales as $R_{\text{aw}} \propto \rho g B^2 H^2 / L$, where $B$ is beam
  and $L$ is length — i.e. quadratic in wave height with a geometric
  hull-shape prefactor.

- **Faltinsen et al. (1980)** extend this to short waves using an asymptotic
  diffraction formulation. Their result also gives $R_{\text{aw}} \propto H^2$
  and adds an explicit heading dependence through the encounter angle $\beta$.

- **ITTC (2014)** recommends the simplified Stawave-2 formula:
  $$
  R_{\text{aw}} = \frac{1}{16} \, \rho \, g \, H^2 \, B \, \sqrt{B/L}
  $$
  which is speed-independent (resistance does not depend on $v$), so power
  grows as $P = R_{\text{aw}} \cdot v \propto H^2 \cdot v^1$.

All three frameworks agree on the $H^2$ dependence (a direct consequence of
linear wave theory, where wave energy $\propto H^2$). They differ on the speed
dependence _of the resistance_:

| Framework                           | $R_{\text{aw}}$ speed dependence                              | $P_{\text{wave}}$ speed dependence      |
| ----------------------------------- | ------------------------------------------------------------- | --------------------------------------- |
| ITTC Stawave-2                      | $\propto v^0$                                                 | $\propto H^2 \cdot v^1$                 |
| Gerritsma-Beukelman (motions-based) | $\propto v^{0.5\text{--}1}$ (varies with encounter frequency) | $\propto H^2 \cdot v^{1.5\text{--}2}$   |
| Experimental regressions            | $\propto v^{0.5\text{--}1.5}$                                 | $\propto H^2 \cdot v^{1.5\text{--}2.5}$ |

The SWOPP3 model uses $P_{\text{wave}} \propto v^{3/2}$, which implies a
resistance that grows as $R_{\text{aw}} \propto v^{1/2}$. This sits between
the speed-independent ITTC approximation and the motions-based strip theory
results — a physically natural compromise. The $3/2$ exponent is best
understood as a regression fit that captures the average speed sensitivity
across the operating envelope, rather than a first-principles derivation.

Strong directional decay from head to following seas is also consistent with
all three frameworks: added resistance is maximum in head seas and negligible
in following seas, where the ship rides with the wave crests.

**References:**

- Gerritsma, J., Beukelman, W. (1972). "Analysis of the resistance increase in waves of a fast cargo ship." _International Shipbuilding Progress_, 19(217), 285–293. [doi:10.3233/ISP-1972-1921701](https://journals.sagepub.com/doi/abs/10.3233/ISP-1972-1921701)
- Salvesen, N. (1978). "Added resistance of ships in waves." _Journal of Hydronautics_, 12(1), 24–34. [doi:10.2514/3.63110](https://doi.org/10.2514/3.63110)
- Faltinsen, O.M., Minsaas, K.J., Liapis, N., Skjørdal, S.O. (1980). "Prediction of resistance and propulsion of a ship in a seaway." _Proc. 13th Symposium on Naval Hydrodynamics_, Tokyo, 505–529. ([Conference proceedings, no DOI available](https://scispace.com/pdf/prediction-of-resistance-and-propulsion-of-a-ship-in-a-lzk79rkb4j.pdf))

### Accuracy

Tested against the reference binary on 10,000 random inputs:

| Metric              | Value    |
| ------------------- | -------- |
| Mean absolute error | 0.008 kW |
| p99 absolute error  | 0.064 kW |
| Max absolute error  | 0.114 kW |
| Max relative error  | 0.031%   |

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

**Derivation of $K_s$:** The value 0.85903125 was identified by isolating the
sail contribution from the binary. Setting $\text{AWA}$ to a known angle where
$\sin\alpha(1 + \frac{3}{20}\sin^2\alpha)$ evaluates to a clean value and
solving for $K_s = P_{\text{sail}} / (V_R^2 \cdot v \cdot C(\alpha))$ yields
the decimal 0.85903125 exactly. This is a terminating decimal:

$$
0.85903125 = \frac{85903125}{10^8} = \frac{27489}{32000}
$$

In other words, $K_s = 27489/32000$ — a ratio of relatively small integers,
suggesting it was likely chosen analytically rather than fitted numerically.

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

   $$
   K_s = \frac{1}{2} \cdot \rho_{\text{air}} \cdot C_L \cdot A_{\text{sail}} \cdot \frac{1}{1000}
   $$

   With total sail area $A_{\text{sail}} = 4 \times 138 = 552 \;\text{m}^2$:

   $$
   C_L = \frac{K_s}{\frac{1}{2} \cdot 1.225 \cdot 552 \cdot 10^{-3}} \approx 2.54
   $$

   This effective lift coefficient is within the typical range $2\text{--}4$
   for rigid wingsails.

### Accuracy

The closed-form sail model is **exact to machine precision** ($< 10^{-13}$)
against the reference. The only residual error in the full model comes from
the $A_w$ constant in the wave term.

Tested against the reference binary on 50,000 random inputs:

| Metric              | Value      |
| ------------------- | ---------- |
| Mean absolute error | 0.004 kW   |
| p99 absolute error  | 0.030 kW   |
| Max absolute error  | 0.050 kW   |
| Errors > 0.1 kW     | 0 / 50,000 |

---

## Summary of Constants

| Constant                  | Symbol | Value       | Exact                           |
| ------------------------- | ------ | ----------- | ------------------------------- |
| Hull coefficient          | $K_h$  | 4.28761…    | $969/226$                       |
| Air drag coefficient      | $K_a$  | 0.153125    | $49/320$                        |
| Wave amplitude            | $A_w$  | 11.1395     | fitted                          |
| Wave directional decay    | $K_w$  | 0.28935…    | $125/432 = 5^3/(2^4 \cdot 3^3)$ |
| Sail thrust coefficient   | $K_s$  | 0.85903125  | exact                           |
| Sail dead zone angle      | —      | 10°         | exact                           |
| Sail quadratic correction | —      | 3/20 = 0.15 | exact                           |

## Physical Coefficients (88 m Cargo Vessel)

| Coefficient      | Symbol    | Derived Value | Typical Range | Source                                                              |
| ---------------- | --------- | ------------- | ------------- | ------------------------------------------------------------------- |
| Total resistance | $C_T$     | 0.0038        | 0.002–0.005   | $K_h / (\frac{1}{2} \rho_w S / 1000)$, $S \approx 2200\;\text{m}^2$ |
| Wind drag × area | $C_D A_T$ | 250 m²        | —             | $K_a / (\frac{1}{2} \rho_a / 1000)$                                 |
| Wind drag        | $C_D$     | ~0.7          | 0.6–0.9       | Assuming $A_T \approx 357\;\text{m}^2$                              |
| Wingsail lift    | $C_L$     | 2.54          | 2–4           | $K_s / (\frac{1}{2} \rho_a \cdot 552 / 1000)$                       |

Where $\rho_w = 1025 \;\text{kg/m}^3$, $\rho_a = 1.225 \;\text{kg/m}^3$.

### Wave Dominance

At typical service speed ($v = 8\;\text{kn}$), wave-added resistance exceeds
hull resistance at surprisingly low wave heights:

| SWH (m) | $P_{\text{wave}}$ / $P_{\text{hull}}$ | $P_{\text{wave}}$ share |
| ------- | ------------------------------------- | ----------------------- |
| 1.0     | 0.31                                  | 24%                     |
| 1.8     | 1.00                                  | 50% (crossover)         |
| 3.0     | 2.78                                  | 74%                     |
| 4.0     | 4.98                                  | 83%                     |

(Head seas, $\text{mwa} = 0°$.)
This is consistent with the SWOPP3 binary itself — wave dominance
is a genuine physical property of the model, not a modelling artifact.

## Clamping Rule

Both functions clamp the result at zero:

$$
P = \max(0, P_{\text{raw}})
$$

This occurs when strong tailwinds ($P_{\text{wind}} < 0$) or large sail
savings ($P_{\text{sail}}$) exceed hull + wave resistance.
