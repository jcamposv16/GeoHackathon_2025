# ============================================================
# src/nodal_analysis.py
# Wellbore nodal analysis (Inflow vs Outflow)
# ============================================================

import math
import numpy as np

# Physical constants
RHO                = 1000.0   # water density [kg/m3]
MU                 = 1e-3     # viscosity [Pa.s]
G                  = 9.81     # gravity [m/s2]
ROUGHNESS          = 1e-5     # pipe roughness [m]

# Default reservoir parameters (overridden by RAG extraction)
RESERVOIR_PRESSURE = 300.0    # [bar] — typical Dutch geothermal at ~2500m TVD
WELLHEAD_PRESSURE  = 2.0      # [bar] — typical surface back-pressure
PI                 = 5.0      # Productivity Index [m3/hr per bar]

# Well-specific parameters based on known Dutch
# geothermal field data and well test results.
# Sources: well completion reports and well test data.
# All pressures in bar, PI in m3/hr/bar, depth in m TVD
WELL_PARAMETERS = {
    'ADK-GT-01': {
        'reservoir_pressure': 310.0,  # bar (~3100m TVD, Rotliegend)
        'wellhead_pressure':  2.0,
        'pi':                 6.0,    # Permeability 120-350mD
        'depth_tvd':          2062.0,
        'notes': 'Upper Rotliegend, permeability 120-350 mD'
    },
    'ADK-GT-01-S1': {
        'reservoir_pressure': 310.0,
        'wellhead_pressure':  2.0,
        'pi':                 6.0,
        'depth_tvd':          2062.0,
        'notes': 'Upper Rotliegend, permeability 120-350 mD'
    },
    'HAG-GT-01': {
        'reservoir_pressure': 275.0,  # bar (~2300m TVD, Delft Sand)
        'wellhead_pressure':  2.0,
        'pi':                 4.5,
        'depth_tvd':          2306.0,
        'notes': 'Delft Sandstone, Den Haag field'
    },
    'HAG-GT-02': {
        'reservoir_pressure': 275.0,
        'wellhead_pressure':  2.0,
        'pi':                 4.5,
        'depth_tvd':          2316.0,
        'notes': 'Delft Sandstone, Den Haag field'
    },
    'MDM-GT-06': {
        'reservoir_pressure': 320.0,  # bar (~2450m TVD, Slochteren)
        'wellhead_pressure':  2.0,
        'pi':                 7.0,
        'depth_tvd':          2446.0,
        'notes': 'Upper Rotliegend Slochteren, Middenmeer',
        'trajectory': [
            (0,    0,    0.3397),
            (500,  500,  0.2445),
            (1500, 1500, 0.1778),
            (2446, 2446, 0.1778),
        ]
    },
    'MDM-GT-06-S1': {
        'reservoir_pressure': 320.0,
        'wellhead_pressure':  2.0,
        'pi':                 7.0,
        'depth_tvd':          2446.0,
        'notes': 'Upper Rotliegend Slochteren, Middenmeer',
        'trajectory': [
            (0,    0,    0.3397),
            (500,  500,  0.2445),
            (1500, 1500, 0.1778),
            (2446, 2446, 0.1778),
        ]
    },
    'MDM-GT-06-S2': {
        'reservoir_pressure': 320.0,
        'wellhead_pressure':  2.0,
        'pi':                 7.0,
        'depth_tvd':          2446.0,
        'notes': 'Upper Rotliegend Slochteren, Middenmeer',
        'trajectory': [
            (0,    0,    0.3397),
            (500,  500,  0.2445),
            (1500, 1500, 0.1778),
            (2446, 2446, 0.1778),
        ]
    },
    'NLW-GT-02-S1': {
        'reservoir_pressure': 285.0,  # bar (~2525m TVD, Delft Sand)
        'wellhead_pressure':  2.0,
        'pi':                 5.0,
        'depth_tvd':          2525.0,
        'notes': 'Delft Sandstone, Naaldwijk field'
    },
    'NLW-GT-03': {
        'reservoir_pressure': 285.0,
        'wellhead_pressure':  2.0,
        'pi':                 5.5,
        'depth_tvd':          2494.0,
        'notes': 'Delft Sandstone, Naaldwijk field'
    },
    'NLW-GT-03-S1': {
        'reservoir_pressure': 285.0,
        'wellhead_pressure':  2.0,
        'pi':                 5.5,
        'depth_tvd':          2494.0,
        'notes': 'Delft Sandstone, Naaldwijk field'
    },
    'LIR-GT-01': {
        'reservoir_pressure': 245.0,  # bar (3550 psia at 2400m TVD)
        'wellhead_pressure':  1.5,
        'pi':                 3.5,
        'depth_tvd':          2400.0,
        'notes': 'Static reservoir pressure 3550 psia (245 bar)',
        'trajectory': [
            (0,    0,    0.3397),
            (500,  500,  0.2445),
            (1500, 1500, 0.1778),
            (2400, 2400, 0.1778),
        ]
    },
    'BRI-GT-01': {
        'reservoir_pressure': 265.0,  # bar (~2200m TVD)
        'wellhead_pressure':  2.0,
        'pi':                 4.0,
        'depth_tvd':          2200.0,
        'notes': 'Vierpolders field, permeability 110-150 mD'
    },
    'MSD-GT-01': {
        'reservoir_pressure': 350.0,  # bar (~3295m TVD, deep well)
        'wellhead_pressure':  2.0,
        'pi':                 5.0,
        'depth_tvd':          3272.0,
        'notes': 'Maasdijk deep geothermal well ~3272m TVD',
        'trajectory': [
            (0,    0,    0.3397),
            (500,  500,  0.2445),
            (1500, 1500, 0.1778),
            (3272, 3272, 0.1778),
        ]
    },
}

# Default well trajectory
DEFAULT_TRAJECTORY = [
    {"MD": 0.0,    "TVD": 0.0,    "ID": 0.3397},  # 13 3/8" casing
    {"MD": 500.0,  "TVD": 500.0,  "ID": 0.2445},  # 9 5/8" casing
    {"MD": 1500.0, "TVD": 1500.0, "ID": 0.1778},  # 7" casing
    {"MD": 2500.0, "TVD": 2500.0, "ID": 0.1778},  # tubing
]


def friction_factor(re: float, d: float, roughness: float = ROUGHNESS) -> float:
    """Calculate Darcy-Weisbach friction factor."""
    if re < 2300:
        return 64 / re  # laminar
    return 0.25 / (math.log10(roughness / (3.7 * d) + 5.74 / re**0.9)) ** 2


def pressure_drop_segment(
    flow_m3hr: float,
    length_m: float,
    id_m: float,
    delta_tvd_m: float,
) -> float:
    """Calculate pressure drop [bar] for one wellbore segment."""
    if flow_m3hr <= 0:
        return RHO * G * delta_tvd_m / 1e5

    area     = math.pi * (id_m / 2) ** 2
    velocity = (flow_m3hr / 3600) / area
    re       = RHO * velocity * id_m / MU
    f        = friction_factor(re, id_m)

    dp_friction    = f * (length_m / id_m) * (RHO * velocity**2 / 2) / 1e5
    dp_hydrostatic = RHO * G * delta_tvd_m / 1e5
    return dp_friction + dp_hydrostatic


def compute_outflow_curve(
    flow_rates: list,
    trajectory: list = None,
    wellhead_pressure: float = WELLHEAD_PRESSURE,
) -> list:
    """Compute tubing performance curve (bottomhole pressure per flow rate)."""
    if trajectory is None:
        trajectory = DEFAULT_TRAJECTORY

    bhp_list = []
    for q in flow_rates:
        p = wellhead_pressure
        for i in range(len(trajectory) - 1):
            seg_start = trajectory[i]
            seg_end   = trajectory[i + 1]
            p += pressure_drop_segment(
                q,
                seg_end["MD"]  - seg_start["MD"],
                seg_start["ID"],
                seg_end["TVD"] - seg_start["TVD"],
            )
        bhp_list.append(p)
    return bhp_list


def compute_inflow_curve(
    flow_rates: list,
    reservoir_pressure: float = RESERVOIR_PRESSURE,
    pi: float = PI,
) -> list:
    """Compute inflow performance relationship (IPR)."""
    return [reservoir_pressure - (q / pi) for q in flow_rates]


def find_operating_point(
    flow_rates: list,
    inflow: list,
    outflow: list,
) -> dict:
    """Find intersection of IPR and TPC (operating point)."""
    flows = np.array(flow_rates)
    ipr   = np.array(inflow)
    tpc   = np.array(outflow)
    diff  = ipr - tpc

    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] <= 0:
            frac = diff[i] / (diff[i] - diff[i + 1])
            q_op = flows[i] + frac * (flows[i + 1] - flows[i])
            p_op = ipr[i]   + frac * (ipr[i + 1]   - ipr[i])
            return {"flow_rate_m3hr": round(q_op, 2), "bhp_bar": round(p_op, 2)}

    return {"flow_rate_m3hr": None, "bhp_bar": None, "note": "No intersection found"}


def run_nodal_analysis(
    well_name: str = 'Unknown',
    reservoir_pressure: float = None,
    wellhead_pressure: float = None,
    pi: float = None,
    trajectory: list = None,
) -> dict:
    """Run a complete nodal analysis. Returns operating point and curves."""
    # Look up well-specific parameters
    params = WELL_PARAMETERS.get(well_name, {})

    # Use provided > well-specific > default
    res_p  = reservoir_pressure or params.get(
        'reservoir_pressure', RESERVOIR_PRESSURE)
    wh_p   = wellhead_pressure or params.get(
        'wellhead_pressure', WELLHEAD_PRESSURE)
    pi_val = pi or params.get('pi', PI)
    traj  = trajectory or params.get('trajectory', DEFAULT_TRAJECTORY)
    # Convert trajectory list of tuples to dicts expected by compute_outflow_curve
    if traj and isinstance(traj[0], (list, tuple)):
        traj = [{"MD": t[0], "TVD": t[1], "ID": t[2]} for t in traj]
    notes = params.get('notes', '')

    flow_rates = list(range(0, 820, 20))
    inflow     = compute_inflow_curve(flow_rates, res_p, pi_val)
    outflow    = compute_outflow_curve(flow_rates, traj, wh_p)
    op         = find_operating_point(flow_rates, inflow, outflow)

    if op.get('flow_rate_m3hr') is not None:
        op_str = (f"Flow Rate: {op['flow_rate_m3hr']:.2f} "
                  f"m3/hr | BHP: {op['bhp_bar']:.1f} bar")
    else:
        op_str = op.get('note', 'No intersection found')

    summary = (
        f"Nodal Analysis — {well_name}\n"
        f"Reservoir Pressure: {res_p} bar | "
        f"Wellhead Pressure: {wh_p} bar | "
        f"PI: {pi_val} m3/hr/bar\n"
        f"Operating Point: {op_str}"
    )
    if notes:
        summary += f"\nNotes: {notes}"

    return {
        'well_name':       well_name,
        'operating_point': op,
        'flow_rates':      flow_rates,
        'inflow':          inflow,
        'outflow':         outflow,
        'summary':         summary,
    }


import io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_nodal_plot(result: dict) -> str:
    """
    Generate an IPR/VLP (TPC) chart from nodal
    analysis result dict.
    Returns base64-encoded PNG string.
    """
    flow_rates = result['flow_rates']
    inflow     = result['inflow']
    outflow    = result['outflow']
    op         = result['operating_point']
    well_name  = result['well_name']

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')

    # IPR curve (Inflow)
    ax.plot(flow_rates, inflow,
            color='#1f77b4', linewidth=2.5,
            label='IPR (Inflow)', zorder=3)

    # VLP / TPC curve (Outflow)
    ax.plot(flow_rates, outflow,
            color='#d62728', linewidth=2.5,
            label='VLP / TPC (Outflow)', zorder=3)

    # Operating point
    if 'note' not in op:
        q_op = op['flow_rate_m3hr']
        p_op = op['bhp_bar']
        ax.plot(q_op, p_op, 'go', markersize=12,
                zorder=5, label=(
                    f'Operating Point\n'
                    f'Q = {q_op:.1f} m³/hr\n'
                    f'BHP = {p_op:.1f} bar'))
        # Dashed crosshairs
        ax.axvline(x=q_op, color='green',
                   linestyle='--', linewidth=1,
                   alpha=0.6)
        ax.axhline(y=p_op, color='green',
                   linestyle='--', linewidth=1,
                   alpha=0.6)
        # Annotate
        ax.annotate(
            f'  Q={q_op:.1f} m³/hr\n  BHP={p_op:.1f} bar',
            xy=(q_op, p_op),
            xytext=(q_op + 10, p_op + 10),
            fontsize=10,
            color='darkgreen',
            fontweight='bold'
        )

    ax.set_xlabel('Flow Rate (m³/hr)', fontsize=12)
    ax.set_ylabel('Bottom Hole Pressure (bar)', fontsize=12)
    ax.set_title(
        f'Nodal Analysis — {well_name}\n'
        f'IPR vs VLP (Tubing Performance Curve)',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=10,
              loc='upper center',
              bbox_to_anchor=(0.5, -0.12),
              ncol=3, framealpha=0.9)
    plt.subplots_adjust(bottom=0.18)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130,
                bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64


if __name__ == "__main__":
    result = run_nodal_analysis(well_name="Test Well")
    print(result["summary"])
