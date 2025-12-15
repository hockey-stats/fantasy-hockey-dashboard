import os
import json
import zipfile
from datetime import datetime, timedelta
import requests
import streamlit as st
import polars as pl
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode


st.set_page_config(layout='wide')

## Constants #########################################################################
SKATER_POSITIONS = ['C', 'LW', 'RW', 'D', 'F', 'All Skaters']
GOALIE_POSITIONS = ['G']
ALL_POSITIONS = SKATER_POSITIONS + GOALIE_POSITIONS
ACCENT = "teal"

# Define minimum thresholds for players to meet to be displayed over a given term
THRESHOLDS = {
    "All Batters": {
        "week": 15,
        "month": 50,
        "season": 100
    },
    "SP": {
        "week": 6,
        "month": 10,
        "season": 80
    },
    "RP": {
        "week": 1,
        "month": 10,
        "season": 25
    },
    "All Pitchers": {
        "week": 1,
        "month": 10,
        "season": 25
    }
}
## End Constants #####################################################################


#@st.cache_data
def load_data(today: str) -> None:
    """
    Function to be run at the initialization of the dashboard.

    Downloads all of the relevant CSV data files from GitHub, where they are stored as build
    artifact for a build that runs daily and scrapes the relevant statistics.

    Expects GitHub PAT with proper permissions to be available as an environment variable under
    'GITHUB_PAT'.

    :param str date: The date we want the data for, in YYYY-mm-dd format, which will usually be 
                     today. Added as a parameter for caching purposes.

    :raises ValueError: Raises a ValueError if an artifact with today's timestamp is not found.
    """

    url = "https://api.github.com/repos/hockey-stats/fantasy-hockey-dashboard/actions/artifacts"
    payload = {}
    headers = {
        'Authorization': f'Bearer {os.environ["GITHUB_PAT"]}'
    }
    output_filename = 'data.zip'
    # Returns a list of every available artifact for the repo
    response = requests.request("GET", url, headers=headers, data=payload, timeout=10)
    response_body = json.loads(response.text)

    print(response_body)
    for artifact in response_body['artifacts']:
        if artifact['name'] == 'nhl-dashboard-fa-data':
            print(artifact['created_at'])
            artifact_creation_date = artifact['created_at'].split('T')[0]
            if today == artifact_creation_date:
                download_url = artifact['archive_download_url']
                break
                # Breaks when we find an artifact with the correct name and today's date
    else:
        # Raise an error if no such artifact as found
        raise ValueError(f"Data for {today} not found, exiting....")

    print(f"Found artifact at {download_url}")
    print("Downloading...")

    # Downloads the artifact as a zip file...
    dl_response = requests.request("GET", download_url, headers=headers, data=payload, timeout=20)
    with open(output_filename, 'wb') as fo:
        fo.write(dl_response.content)

    print("Download complete")

    # ... and unzips
    with zipfile.ZipFile(output_filename, 'r') as zip_ref:
        zip_ref.extractall('data')

    print(os.listdir('data'))

    print(f"Data loaded for {artifact_creation_date}")


########################################################################################
## Main Script #########################################################################
########################################################################################

# Get data for today's date.
today = datetime.today()
# If checking before 7am UTC, use yesterday's data instead, since data hasn't been updated yet
if today.hour <= 7:
    today -= timedelta(days=1)
load_data(today.strftime('%Y-%m-%d'))

# Set title
st.markdown(
    """
    # Interesting Free Agents (NHL Fantasy)
    """
)

# Get two columns for our page
l_column, r_column = st.columns([0.63, 0.37])

# Add the position selector to left column...
with l_column:
    chosen_position = st.selectbox(
        label="Position:",
        options=ALL_POSITIONS,
    )

# ... and term selector on the right
with r_column:
    chosen_term = st.radio(
        label="Chosen term:",
        options=['Last Week', 'Last Month', 'Full Season'],
        index=0,
        horizontal=True
    )

# Load the correct CSV for chosen position
if chosen_position in SKATER_POSITIONS:
    df = pl.read_csv("data/skater_data.csv")
    if chosen_position == 'F':
        df = df.filter(pl.col('Position(s)').str.contains('C|RW|LW'))
    elif chosen_position != 'All Skaters':
        df = df.filter(pl.col('Position(s)').str.contains(chosen_position))

elif chosen_position in GOALIE_POSITIONS:
    df = pl.read_csv("data/goalie_data.csv")


########################################################################################
##  Begin Table ########################################################################
########################################################################################

term = chosen_term.split(' ')[-1].lower()
table_df = df.filter(pl.col('term') == term)

DISPLAY_NUMBER = 25 if 'All' in chosen_position else 15

table_df = table_df.sort(by=['on_team', 'Rank'], descending=[True, False]).head(DISPLAY_NUMBER)

if chosen_position in SKATER_POSITIONS:
    table_df = table_df[['Name', 'Team', 'Position(s)', 'G', 'A', '+/-', 'PPP', 'SOG', 'HITS',
                         'Rank', 'ixG', 'EV ToI/g', 'PP ToI/g', 'on_team']]
else:
    table_df = table_df[['Name', 'Team', 'Position(s)', 'W', 'GA', 'GAA', 'SV%', 'SHO', 'Rank',
                         'GSAX/g', 'xGA/g', 'GP', 'on_team']]

table_df = table_df.rename({'Position(s)': 'Pos.'})

# Format names, e.g. Auston Matthews -> A. Matthews
table_df = table_df.with_columns(
    pl.col('Name').map_elements(lambda x: f"{x[0]}. {' '.join(x.split(' ')[1:])}",
                                return_dtype=pl.String)
)

small_cols = ['G', 'A', '+/-', 'PPP', 'SOG', 'HITS', 'Rank', 'W', 'GA', 'SHO', 'ixG',
              'EV ToI/g', 'PP ToI/g', 'GSAX/g', 'xGA/g', 'GP', 'Team', 'Pos.']

# Define column options for each column we want to include
columnDefs = [
    {
    'field': col,
    'headerName': col,
    'type': 'rightAligned',
    'width': 10 if col in small_cols else 40,
    'height': 20,
    'sortable': True,
    'sortingOrder': ['desc', 'asc', None]
    } for col in list(table_df.columns) if col != 'on_team'
]

# Format the decimal numbers for certain metrics
for colDef in columnDefs:
    if colDef['field'] in {'GAA', 'SV%', 'ixG', 'EV ToI/g', 'PP ToI/g',
                           'GSAX/g', 'xGA/g'}:
        colDef['type'] = ['numericColumn', 'customNumericFormat']
        colDef['precision'] = 2

# Set the name column (always the first one) to be left-aligned
columnDefs[0]['type'] = 'leftAligned'
columnDefs[0]['width'] = 60

with l_column:
    # Define CSS rule to color the rows for every player on our team.
    cellStyle = JsCode(
        r"""
        function(cellClassParams) {
            if (cellClassParams.data.on_team) {
                return {'background-color': '#a6761d'}
            }
            return {};
        }
        """)

    # Define the font size for the table
    css = {
            ".ag-row": {"font-size": "12pt"},
            ".ag-header": {"font-size": "12pt"}
        }

    grid_builder = GridOptionsBuilder.from_dataframe(table_df)
    grid_options = grid_builder.build()

    # Add the cell style rule to each column
    grid_options['defaultColDef']['cellStyle'] = cellStyle
    # Set height/width of columns automatically
    grid_options['defaultColDef']['autoHeight'] = True
    grid_options['defaultColDef']['autoWidth'] = True

    grid_options['columnDefs'] = columnDefs

    # Add the table to our dashboard
    AgGrid(table_df, gridOptions=grid_options, allow_unsafe_jscode=True,
           fit_columns_on_grid_load=True, custom_css=css,
           height=485)

########################################################################################
##  End Table ##########################################################################
########################################################################################

########################################################################################
## End Main Script #####################################################################
########################################################################################
