import polars as pl
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

from update_data import build_free_agent_data

## Constants #########################################################################
SKATER_POSITIONS = ['C', 'LW', 'RW', 'D', 'F', 'All Skaters']
GOALIE_POSITIONS = ['G']
ALL_POSITIONS = SKATER_POSITIONS + GOALIE_POSITIONS
## End Constants #####################################################################


@st.cache_data(ttl=6 * 3600, show_spinner='Pulling player stats from Yahoo (~1 min)...')
def load_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Wrapper around the Yahoo data pull to use caching."""
    return build_free_agent_data()


skater_df, goalie_df = load_data()

st.markdown(
    """
    # Interesting Free Agents (NHL Fantasy)
    """
)

l_column, r_column = st.columns([0.95, 0.05])

with l_column:
    chosen_position = st.selectbox(
        label="Position:",
        options=ALL_POSITIONS,
    )

    chosen_term = st.radio(
        label="Chosen term:",
        options=['Last Week', 'Last Month', 'Full Season'],
        index=0,
        horizontal=True
    )

if chosen_position in SKATER_POSITIONS:
    df = skater_df
    if chosen_position == 'F':
        df = df.filter(pl.col('Position(s)').str.contains('C|RW|LW'))
    elif chosen_position != 'All Skaters':
        df = df.filter(pl.col('Position(s)').str.contains(chosen_position))
else:
    df = goalie_df


########################################################################################
##  Begin Table ########################################################################
########################################################################################

term = chosen_term.split(' ')[-1].lower()
table_df = df.filter(pl.col('term') == term)

DISPLAY_NUMBER = 25 if chosen_position in {'All Skaters', 'F'} else 15

table_df = table_df.sort(by=['on_team', 'Rank'], descending=[True, False]).head(DISPLAY_NUMBER)

if chosen_position in SKATER_POSITIONS:
    table_df = table_df[['Name', 'Team', 'Position(s)', 'G', 'A', '+/-', 'PPP', 'SOG', 'HITS',
                          'Rank', 'on_team']]
else:
    table_df = table_df[['Name', 'Team', 'Position(s)', 'W', 'GA', 'GAA', 'SV%', 'SHO', 'Rank',
                          'on_team']]

table_df = table_df.rename({'Position(s)': 'Pos.'})

# Format names, e.g. Auston Matthews -> A. Matthews
table_df = table_df.with_columns(
    pl.col('Name').map_elements(lambda x: f"{x[0]}. {' '.join(x.split(' ')[1:])}",
                                return_dtype=pl.String)
)

small_cols = ['G', 'A', '+/-', 'PPP', 'SOG', 'HITS', 'Rank', 'W', 'GA', 'SHO', 'Team', 'Pos.']

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

# Format the decimal numbers for goalie rate stats
for colDef in columnDefs:
    if colDef['field'] in {'GAA', 'SV%'}:
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
