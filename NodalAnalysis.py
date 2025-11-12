import math
import numpy as np
import matplotlib.pyplot as plt

# %% Hardcoded parameters

rho = 1000.0                 # water density [kg/m3]
mu = 1e-3                    # viscosity [Pa.s]
g = 9.81                     # gravity [m/s2]
roughness = 1e-5             # pipe roughness [m]
reservoir_pressure = 230.0   # bar
wellhead_pressure = 10.0     # bar
PI = 5.0                     # Productivity Index [m3/hr per bar]
esp_depth = 500.0            # ESP intake depth [m]

# ESP pump curve
pump_curve = {
    "flow": [0, 100, 200, 300, 400],     # m3/hr
    "head": [600, 550, 450, 300, 100],   # m
}

# %% Well trajectory
well_trajectory = [
    {"MD": 0.0,    "TVD": 0.0,    "ID": 0.3397},  # 13 3/8" casing, MD, TVD, ID are in meters
    {"MD": 500.0,  "TVD": 500.0,  "ID": 0.2445},  # 9 5/8" casing, MD, TVD, ID are in meters
    {"MD": 1500.0, "TVD": 1500.0, "ID": 0.1778},  # 7" casing, MD, TVD, ID are in meters
    {"MD": 2500.0, "TVD": 2500.0, "ID": 0.1778},  # tubing section, MD, TVD, ID are in meters
]

# Well segments
segments = []
for i in range(1, len(well_trajectory)):
    MD = well_trajectory[i]["MD"] - well_trajectory[i-1]["MD"]
    TVD = well_trajectory[i]["TVD"] - well_trajectory[i-1]["TVD"]
    D = well_trajectory[i]["ID"]
    L = MD
    theta = math.atan2(TVD, MD)
    segments.append((L, D, theta))

# %% Calculation functions

# Friction 
def swamee_jain(Re, D):
    if Re <= 0: return 0.0
    return 0.25 / (math.log10((roughness/(3.7*D)) + (5.74/(Re**0.9))))**2

# Pump head 
def pump_interp(flow, key):
    return np.interp(flow, pump_curve["flow"], pump_curve[key])

# VLP 
def vlp(flow_m3hr):
    q = flow_m3hr / 3600.0  # m3/hr to m3/s
    dp_total = 0.0
    depth_accum = 0.0
    for (L, D, theta) in segments:
        A = math.pi * D**2 / 4.0
        u = q / A
        Re = rho * abs(u) * D / mu
        f = swamee_jain(Re, D)

        dp_fric = f * (L/D) * (rho * u**2 / 2.0)
        dp_grav = rho * g * L * math.sin(theta)
        dp_total += dp_fric + dp_grav
        depth_accum += L * math.sin(theta)

    if depth_accum >= esp_depth:
        dp_total -= rho * g * pump_interp(flow_m3hr, "head")

    return wellhead_pressure + dp_total/1e5

# IPR 
def ipr(flow_m3hr):
    pbh = reservoir_pressure - flow_m3hr / PI
    return max(pbh, 0.0)


# %% Results

flows = np.linspace(1, 400, 200)
p_vlp = np.array([vlp(f) for f in flows])
p_ipr = np.array([ipr(f) for f in flows])

# Find solution: minimum difference
diff = np.abs(p_vlp - p_ipr)
idx = np.argmin(diff)

if diff[idx] < 3:  # tolerance in bar
    sol_flow = flows[idx]
    sol_pbh = p_vlp[idx]
    sol_head = pump_interp(sol_flow, "head")
else:
    sol_flow = sol_pbh = sol_head = None

# Plot results
plt.figure()
plt.plot(flows, p_vlp, label="VLP")
plt.plot(flows, p_ipr, label="IPR")
if sol_flow:
    plt.scatter(sol_flow, sol_pbh, color="red",
                label=f"Operating point Q={sol_flow:.1f} m3/hr, BHP={sol_pbh:.1f} bar")
plt.xlabel("Flowrate [m3/hr]")
plt.ylabel("Pressure [bar]")
plt.legend()
plt.grid()
plt.show()

# Print solution
if sol_flow:
    print("Solution found:")
    print(f"Flowrate: {sol_flow:.2f} m3/hr")
    print(f"Bottomhole pressure: {sol_pbh:.2f} bar")
    print(f"Pump head: {sol_head:.1f} m")
else:
    print("No solution found with current settings.")
