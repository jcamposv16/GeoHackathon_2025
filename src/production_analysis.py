import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io, base64
from pathlib import Path

WELLS_BASE = Path("GeoHackathon_2025/Wells")

# Map well IDs to their production data files
PRODUCTION_FILES = {
    'ADK-GT-01':     'Well 1/Production data/ANDIJK-GT-01_2020_2025.xlsx',
    'ADK-GT-01-S1':  'Well 1/Production data/ANDIJK-GT-01_2020_2025.xlsx',
    'HAG-GT-01':     'Well 2/Production data/DEN HAAG-GT-01_2020_2025.xlsx',
    'HAG-GT-02':     'Well 2/Production data/DEN HAAG-GT-01_2020_2025.xlsx',
    'MDM-GT-06':     'Well 3/Production data/MIDDENMEER-GT-06_2020_2025.xlsx',
    'MDM-GT-06-S1':  'Well 3/Production data/MIDDENMEER-GT-06_2020_2025.xlsx',
    'MDM-GT-06-S2':  'Well 3/Production data/MIDDENMEER-GT-06_2020_2025.xlsx',
    'NLW-GT-02-S1':  'Well 4/Production data/NAALDWIJK-GT-02_2020_2025.xlsx',
    'NLW-GT-03':     'Well 5/Production data/NAALDWIJK-GT-03_2020_2025.xlsx',
    'NLW-GT-03-S1':  'Well 5/Production data/NAALDWIJK-GT-03_2020_2025.xlsx',
}

COLUMN_NAMES = [
    'field', 'well_id', 'operator', 'date',
    'water_produced_m3', 'oil_sm3', 'gas_nm3',
    'condensate_sm3', 'inhibitor_prod_l',
    'water_injected_m3', 'inhibitor_inj_l'
]


def _load_production(well_ids: list) -> pd.DataFrame:
    """Load production data for given well IDs."""
    for wid in well_ids:
        path = PRODUCTION_FILES.get(wid)
        if path:
            full_path = WELLS_BASE / path
            if full_path.exists():
                df = pd.read_excel(
                    str(full_path),
                    sheet_name='Sheet0',
                    header=None,
                    names=COLUMN_NAMES,
                    skiprows=1  # skip Dutch header row
                )
                # Parse date column
                df['date'] = pd.to_datetime(
                    df['date'], format='%Y-%m', errors='coerce')
                # Drop rows with no date
                df = df.dropna(subset=['date'])
                # Convert numeric columns
                for col in ['water_produced_m3', 'gas_nm3',
                            'water_injected_m3']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                return df, wid, full_path.name
    return None, None, None


def generate_production_chart(well_ids: list) -> tuple:
    """
    Generate a production chart for the given wells.
    Returns (summary_text, base64_image_or_none)
    """
    df, matched_id, filename = _load_production(well_ids)

    if df is None or df.empty:
        return ("No production data found for wells: " +
                ', '.join(well_ids), None)

    # Summary statistics
    total_water = df['water_produced_m3'].sum()
    total_gas = df['gas_nm3'].sum()
    avg_monthly = df['water_produced_m3'].mean()
    max_monthly = df['water_produced_m3'].max()
    max_date = df.loc[df['water_produced_m3'].idxmax(), 'date']
    date_range = (
        df['date'].min().strftime('%Y-%m') +
        ' to ' +
        df['date'].max().strftime('%Y-%m')
    )
    n_months = len(df)

    summary = (
        f"Production data for {matched_id} "
        f"({date_range}, {n_months} months):\n"
        f"- Total water produced: {total_water:,.0f} m³\n"
        f"- Average monthly production: {avg_monthly:,.0f} m³/month\n"
        f"- Peak production: {max_monthly:,.0f} m³ "
        f"({max_date.strftime('%Y-%m')})\n"
    )
    if total_gas > 0:
        summary += f"- Total gas: {total_gas:,.0f} Nm³\n"

    # Generate dual-axis line chart
    fig, ax1 = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('white')

    # Left Y-axis — Gas production (Nm³)
    color_gas = '#e07b2a'
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Gas Production (Nm³/month)',
                   fontsize=12, color=color_gas)
    if df['gas_nm3'].notna().any() and df['gas_nm3'].sum() > 0:
        ax1.plot(df['date'], df['gas_nm3'],
                 color=color_gas, linewidth=2,
                 marker='o', markersize=3,
                 label='Gas (Nm³)')
        ax1.tick_params(axis='y', labelcolor=color_gas)
    else:
        ax1.set_visible(False)

    # Right Y-axis — Water production (m³)
    ax2 = ax1.twinx()
    color_water = '#1f77b4'
    ax2.set_ylabel('Water Produced (m³/month)',
                   fontsize=12, color=color_water)
    ax2.plot(df['date'], df['water_produced_m3'],
             color=color_water, linewidth=2,
             marker='o', markersize=3,
             label='Water (m³)')
    ax2.tick_params(axis='y', labelcolor=color_water)

    ax1.set_title(
        f'Monthly Production History — {matched_id}',
        fontsize=14, fontweight='bold'
    )
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    fig.autofmt_xdate(rotation=45, ha='right')

    # Combined legend below plot
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper center',
               bbox_to_anchor=(0.5, -0.18),
               ncol=3, fontsize=10,
               framealpha=0.9)
    plt.subplots_adjust(bottom=0.22)

    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return summary, img_b64


def generate_watercut_chart(well_ids: list) -> tuple:
    """
    Generate a water cut chart (water / total fluid * 100%).
    For geothermal wells water cut is always ~100% but
    we show produced water vs gas equivalent volume trend.
    Returns (summary_text, base64_image_or_none)
    """
    df, matched_id, filename = _load_production(well_ids)

    if df is None or df.empty:
        return ("No production data found for wells: " +
                ', '.join(well_ids), None)

    # For geothermal: water cut = water / (water + gas_volume_equiv)
    # Convert gas Nm3 to m3 liquid equivalent (approx factor 0.001)
    df['gas_m3_equiv'] = df['gas_nm3'].fillna(0) * 0.001
    df['total_fluid'] = df['water_produced_m3'] + df['gas_m3_equiv']
    df['water_cut_pct'] = (
        df['water_produced_m3'] / df['total_fluid'] * 100
    ).where(df['total_fluid'] > 0)

    avg_wc = df['water_cut_pct'].mean()
    min_wc = df['water_cut_pct'].min()
    min_wc_date = df.loc[df['water_cut_pct'].idxmin(), 'date']

    summary = (
        f"Water cut analysis for {matched_id}:\n"
        f"- Average water cut: {avg_wc:.1f}%\n"
        f"- Minimum water cut: {min_wc:.1f}% "
        f"({min_wc_date.strftime('%Y-%m')})\n"
        f"Note: Geothermal wells produce almost pure water. "
        f"Gas fraction is dissolved gas from aquifer.\n"
    )

    # Generate chart
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('white')

    ax.plot(df['date'], df['water_cut_pct'],
            color='#1f77b4', linewidth=2,
            marker='o', markersize=3, label='Water cut %')
    ax.axhline(y=avg_wc, color='red', linewidth=1.5,
               linestyle='--',
               label=f'Average ({avg_wc:.1f}%)')
    ax.fill_between(df['date'], df['water_cut_pct'],
                    alpha=0.15, color='#1f77b4')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Water Cut (%)', fontsize=12)
    ax.set_ylim(
        max(0, df['water_cut_pct'].min() - 5),
        min(100, df['water_cut_pct'].max() + 5)
    )
    ax.set_title(
        f'Water Cut History — {matched_id}',
        fontsize=14, fontweight='bold'
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return summary, img_b64


def generate_cumulative_chart(well_ids: list) -> tuple:
    """Cumulative water and gas production over time."""
    df, matched_id, filename = _load_production(well_ids)
    if df is None or df.empty:
        return ('No production data found for wells: ' +
                ', '.join(well_ids), None)

    df = df.sort_values('date')
    df['cum_water'] = df['water_produced_m3'].cumsum() / 1e6
    df['cum_gas'] = df['gas_nm3'].fillna(0).cumsum() / 1e6

    fig, ax1 = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('white')

    color_water = '#1f77b4'
    color_gas = '#e07b2a'

    ax1.fill_between(df['date'], df['cum_water'],
                     alpha=0.3, color=color_water)
    ax1.plot(df['date'], df['cum_water'],
             color=color_water, linewidth=2,
             label='Cumulative Water (MM m³)')
    ax1.set_ylabel('Cumulative Water (MM m³)',
                   fontsize=12, color=color_water)
    ax1.tick_params(axis='y', labelcolor=color_water)

    ax2 = ax1.twinx()
    ax2.fill_between(df['date'], df['cum_gas'],
                     alpha=0.2, color=color_gas)
    ax2.plot(df['date'], df['cum_gas'],
             color=color_gas, linewidth=2,
             label='Cumulative Gas (MM Nm³)')
    ax2.set_ylabel('Cumulative Gas (MM Nm³)',
                   fontsize=12, color=color_gas)
    ax2.tick_params(axis='y', labelcolor=color_gas)

    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_title(
        f'Cumulative Production — {matched_id}',
        fontsize=14, fontweight='bold')
    ax1.xaxis.set_major_formatter(
        mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(
        mdates.MonthLocator(interval=4))
    fig.autofmt_xdate(rotation=45, ha='right')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper center',
               bbox_to_anchor=(0.5, -0.18),
               ncol=2, fontsize=10)
    plt.subplots_adjust(bottom=0.22)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    total_w = df['water_produced_m3'].sum()
    total_g = df['gas_nm3'].fillna(0).sum()
    summary = (
        f"Cumulative production for {matched_id}:\n"
        f"- Total water: {total_w/1e6:.2f} MM m³\n"
        f"- Total gas: {total_g/1e6:.2f} MM Nm³\n"
    )

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130,
                bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return summary, img_b64


def generate_gas_trend_chart(well_ids: list) -> tuple:
    """Monthly gas production rate trend."""
    df, matched_id, filename = _load_production(well_ids)
    if df is None or df.empty:
        return ('No production data found for wells: ' +
                ', '.join(well_ids), None)

    df = df.sort_values('date')

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('white')

    color_gas = '#e07b2a'
    ax.bar(df['date'], df['gas_nm3'] / 1000,
           color=color_gas, alpha=0.7, width=20,
           label='Monthly Gas (k Nm³)')

    if len(df) > 6:
        rolling = df['gas_nm3'].rolling(6).mean() / 1000
        ax.plot(df['date'], rolling,
                color='darkred', linewidth=2.5,
                linestyle='-',
                label='6-month avg')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Gas Production (k Nm³/month)',
                  fontsize=12)
    ax.set_title(
        f'Gas Production Trend — {matched_id}',
        fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(
        mdates.MonthLocator(interval=4))
    fig.autofmt_xdate(rotation=45, ha='right')
    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.18),
              ncol=2, fontsize=10)
    plt.subplots_adjust(bottom=0.22)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    avg_gas = df['gas_nm3'].mean()
    max_gas = df['gas_nm3'].max()
    max_date = df.loc[df['gas_nm3'].idxmax(), 'date']
    summary = (
        f"Gas production trend for {matched_id}:\n"
        f"- Average monthly gas: {avg_gas:,.0f} Nm³\n"
        f"- Peak gas: {max_gas:,.0f} Nm³ "
        f"({max_date.strftime('%Y-%m')})\n"
    )

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130,
                bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return summary, img_b64


def generate_monthly_comparison_chart(
        well_ids: list) -> tuple:
    """Monthly production grouped by year for
    seasonal pattern analysis."""
    df, matched_id, filename = _load_production(well_ids)
    if df is None or df.empty:
        return ('No production data found for wells: ' +
                ', '.join(well_ids), None)

    df = df.sort_values('date')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    years = sorted(df['year'].unique())
    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May',
                   'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
                   'Nov', 'Dec']

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('white')

    colors = plt.cm.tab10.colors
    width = 0.8 / len(years)

    for i, year in enumerate(years):
        yr_data = df[df['year'] == year]
        monthly = yr_data.groupby('month')[
            'water_produced_m3'].sum() / 1000
        x = [m + i * width - 0.4 for m in months]
        vals = [monthly.get(m, 0) for m in months]
        ax.bar(x, vals, width=width,
               label=str(year),
               color=colors[i % len(colors)],
               alpha=0.8)

    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Water Produced (k m³)', fontsize=12)
    ax.set_title(
        f'Monthly Production Comparison by Year'
        f' — {matched_id}',
        fontsize=14, fontweight='bold')
    ax.set_xticks(list(months))
    ax.set_xticklabels(month_names)
    ax.legend(title='Year',
              loc='upper center',
              bbox_to_anchor=(0.5, -0.12),
              ncol=len(years), fontsize=10)
    plt.subplots_adjust(bottom=0.20)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    summary = (
        f"Monthly production comparison for "
        f"{matched_id} ({len(years)} years):\n"
        f"- Years covered: "
        f"{min(years)} to {max(years)}\n"
        f"- Use this chart to identify seasonal "
        f"patterns in production.\n"
    )

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130,
                bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return summary, img_b64


if __name__ == '__main__':
    summary, img = generate_production_chart(['ADK-GT-01'])
    print(summary)
    if img:
        print('Chart generated successfully (' +
              str(len(img)) + ' bytes base64)')
    else:
        print('No chart generated')
