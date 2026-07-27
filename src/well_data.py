import pandas as pd
from pathlib import Path

XLSX_PATH = Path("GeoHackathon_2025/boreholes.xlsx")

# Column name mapping Dutch -> English
COLUMN_MAP = {
    'Well': 'well_number',
    'Boorgatcode': 'well_id',
    'Boorgatnaam': 'well_name',
    'Putnaam': 'field_well_name',
    'Veldnaam': 'field_name',
    'Boorgatdiepte AH': 'depth_md',
    'Boorgatdiepte TVD': 'depth_tvd',
    'Kickoff diepte AH': 'kickoff_depth',
    'Boorgat / Sidetrack': 'sidetrack_type',
    'Startdatum': 'spud_date',
    'Einddatum': 'end_date',
    'Status': 'status',
    'Huidige eigenaar': 'owner',
    'Opdrachtgever': 'operator',
    'X Rijksdriehoek': 'x_rd',
    'Y Rijksdriehoek': 'y_rd',
    'Latitude WGS84': 'latitude',
    'Longitude WGS84': 'longitude',
    'Gemeentenaam': 'municipality',
    'Boorgatvorm': 'wellbore_profile',
    'Type': 'well_objective',
}

# Manual aliases for IDs that differ between
# hackathon folders and official registry
WELL_ID_ALIASES = {
    'MSD-GT-01':    'MLD-GT-01-S1',
    'MSD-GT-01-S1': 'MLD-GT-01-S1',
    'BRI-GT-01':    'BRI-GT-01-S1',
    'HAG-GT-02':    'HAG-GT-01',
    'MDM-GT-06':    'MDM-GT-06-S2',
    'MDM-GT-06-S1': 'MDM-GT-06-S2',
    'ADK-GT-01':    'ADK-GT-01-S1',
    'NLW-GT-03':    'NLW-GT-03-S1',
}

_df = None


def _load() -> pd.DataFrame:
    global _df
    if _df is None:
        df = pd.read_excel(str(XLSX_PATH), sheet_name='Sheet0')
        df = df.rename(columns=COLUMN_MAP)
        _df = df
    return _df


def get_well_data(well_ids: list) -> dict:
    """
    Return structured well data for the given well IDs.
    Matches on well_id column (Boorgatnaam).
    Returns dict with all available fields.
    """
    df = _load()
    results = {}

    for wid in well_ids:
        # Resolve alias if direct match fails
        lookup_id = WELL_ID_ALIASES.get(wid, wid)

        # Try exact match first
        mask = df['well_id'].str.upper() == lookup_id.upper()
        rows = df[mask]

        if rows.empty:
            # Try partial match
            mask = df['well_id'].str.upper().str.contains(
                lookup_id.upper(), na=False)
            rows = df[mask]

        if not rows.empty:
            row = rows.iloc[0]
            results[wid] = {
                'well_id': row.get('well_id', ''),
                'well_name': row.get('well_name', ''),
                'field_name': row.get('field_name', ''),
                'depth_md': row.get('depth_md', ''),
                'depth_tvd': row.get('depth_tvd', ''),
                'kickoff_depth': row.get('kickoff_depth', ''),
                'is_sidetrack': str(row.get('sidetrack_type', '')) == 'STR',
                'spud_date': str(row.get('spud_date', ''))[:10],
                'end_date': str(row.get('end_date', ''))[:10],
                'status': row.get('status', ''),
                'operator': row.get('operator', ''),
                'owner': row.get('owner', ''),
                'x_rd': row.get('x_rd', ''),
                'y_rd': row.get('y_rd', ''),
                'latitude': row.get('latitude', ''),
                'longitude': row.get('longitude', ''),
                'municipality': row.get('municipality', ''),
                'wellbore_profile': row.get('wellbore_profile', ''),
                'well_objective': row.get('well_objective', ''),
            }

    return results


def format_well_facts(well_ids: list) -> str:
    """
    Return a formatted string of verified well facts
    for injection into LLM prompts.
    """
    data = get_well_data(well_ids)
    if not data:
        return ''

    lines = ['VERIFIED WELL DATA FROM OFFICIAL REGISTRY:']
    for wid, info in data.items():
        lines.append(f'Well ID: {wid}')
        if info['well_name']:
            lines.append(f'  Name: {info["well_name"]}')
        if info['field_name']:
            lines.append(f'  Field: {info["field_name"]}')
        if info['municipality']:
            lines.append(f'  Municipality: {info["municipality"]}')
        if info['operator']:
            lines.append(f'  Operator: {info["operator"]}')
        if info['depth_md'] and str(info['depth_md']) not in ('', 'nan', 'NaN'):
            lines.append(f'  Total Depth MD: {info["depth_md"]}m')
        if info['depth_tvd'] and str(info['depth_tvd']) not in ('', 'nan', 'NaN'):
            lines.append(f'  Total Depth TVD: {info["depth_tvd"]}m')
        if info['spud_date'] and info['spud_date'] != 'NaT':
            lines.append(f'  Spud Date: {info["spud_date"]}')
        if info['is_sidetrack']:
            lines.append(f'  Type: Sidetrack (STR)')
        if info['status']:
            lines.append(f'  Status: {info["status"]}')

    return '\n'.join(lines)
