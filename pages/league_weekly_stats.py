import pandas as pd
import polars as pl
import streamlit as st

import get_league_weekly_stats as lws

st.set_page_config(layout='wide', page_title='League Weekly Stats')

st.markdown('# League Weekly Stats')


@st.cache_data(ttl=3600)
def load_league_weekly_stats() -> tuple[pl.DataFrame, str]:
    """Wrapper to fetch and cache league weekly stats."""
    return lws.run()


league_stats_df, my_team = load_league_weekly_stats()

if league_stats_df.is_empty():
    st.write('League stats data not found. Stats appear once the first week of the season is complete.')
    st.stop()

latest_week = int(league_stats_df.select(pl.col('week').max()).item())

timeframe = st.radio(
    'Select Timeframe:',
    options=['Last 2 Weeks', 'Full Season'],
    horizontal=True,
    index=0,
)

if timeframe == 'Last 2 Weeks':
    st.write(f"**Weeks {max(1, latest_week - 1)} - {latest_week} Average Summary**")
else:
    st.write(f"**Season Average Summary (Weeks 1 - {latest_week})**")

agg_df = lws.get_aggregated_stats(league_stats_df, timeframe)

avg_row = agg_df.select(
    [pl.lit('League Average').alias('team'), *[pl.col(c).mean() for c in lws.CATEGORIES]]
)

# Order: league average first, then my team, then everyone else
ordered_df = pl.concat(
    [
        avg_row,
        agg_df.filter(pl.col('team') == my_team),
        agg_df.filter(pl.col('team') != my_team).sort('team'),
    ],
    how='diagonal',
)

# Transpose so categories are rows and teams are columns
pd_stats = ordered_df.select(['team'] + lws.CATEGORIES).to_pandas().set_index('team').T

# Rank only the actual teams, not the league average
rank_cols = [c for c in pd_stats.columns if c != 'League Average']
higher_is_better = [c for c in lws.CATEGORIES if c not in lws.LOWER_IS_BETTER]

styler = pd_stats.style
styler = styler.background_gradient(
    cmap='RdBu', subset=pd.IndexSlice[lws.LOWER_IS_BETTER, rank_cols], axis=1
)
styler = styler.background_gradient(
    cmap='RdBu_r', subset=pd.IndexSlice[higher_is_better, rank_cols], axis=1
)
styler = styler.format(precision=2, na_rep='-')
styler = styler.format(formatter='{:.3f}', subset=pd.IndexSlice[['SV%'], :])

# st.table renders pandas Styler formatting more reliably than st.dataframe
st.table(styler)
