"""Yahoo-only version of the dashboard, for use while the pyhockey data feed is down.

Pulls everything live from the Yahoo Fantasy API instead of the GitHub Actions data artifact.
Run with: streamlit run main_yahoo.py
"""
import streamlit as st

st.set_page_config(layout='wide')

pages = [
    st.Page('views/free_agents_yahoo.py', title='Interesting Free Agents', icon='📋', default=True),
    st.Page('pages/league_weekly_stats.py', title='League Weekly Stats', icon='📈'),
]

st.navigation(pages, position='top').run()
