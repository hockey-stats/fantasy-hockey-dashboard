import logging

import polars as pl
import yahoo_fantasy_api as yfa

from update_data import create_session, get_league_id

oauth_logger = logging.getLogger('yahoo_oauth')
oauth_logger.disabled = True


# Scoring categories, plus the hidden goalie stats (GA, SV, SA) needed to aggregate GAA and SV%
STAT_MAP = {
    '1': 'G',
    '2': 'A',
    '4': '+/-',
    '8': 'PPP',
    '14': 'SOG',
    '31': 'HIT',
    '19': 'W',
    '22': 'GA',
    '23': 'GAA',
    '24': 'SA',
    '25': 'SV',
    '26': 'SV%',
    '27': 'SHO',
}

CATEGORIES = ['G', 'A', '+/-', 'PPP', 'SOG', 'HIT', 'W', 'GAA', 'SV%', 'SHO']
LOWER_IS_BETTER = ['GAA']


def gather_matchup_stats(matchup: dict, week: int) -> list[dict]:
    """Parses matchup data for the stats for each team.

    Args:
        matchup (dict): Dict object containing info about teams in the matchup
        week (int): The week the matchup took place in

    Returns:
        list[dict]: One dict of stats for each team in the matchup.
    """
    stats_list = []
    teams = matchup['0']['teams']

    for i in ['0', '1']:
        team_data = {'week': week, 'team': None}

        for entry in teams[i]['team'][0]:
            if isinstance(entry, dict) and entry.get('name', False):
                team_data['team'] = entry['name']
                break

        for entry in teams[i]['team'][1]['team_stats']['stats']:
            if stat_name := STAT_MAP.get(entry['stat']['stat_id']):
                try:
                    team_data[stat_name] = float(entry['stat']['value'])
                except (ValueError, TypeError):
                    # Yahoo returns '-' or '' for e.g. GAA when no goalie has played
                    team_data[stat_name] = None

        stats_list.append(team_data)

    return stats_list


def get_league_weekly_stats(league: yfa.League) -> pl.DataFrame:
    """Collects every team's stats for each completed week of the season.

    Args:
        league (yfa.League): The League object to query against

    Returns:
        pl.DataFrame: One row per team per completed week.
    """
    current_week = league.current_week()
    all_stats = []

    for week in range(1, current_week):
        print(f"Fetching stats for week {week}...")
        try:
            matchups = league.matchups(week=week)['fantasy_content']['league'][1]['scoreboard']['0']['matchups']
        except (KeyError, IndexError):
            print(f"No matchups found for week {week}")
            continue

        for matchup_key, matchup in matchups.items():
            if matchup_key == 'count':
                continue
            all_stats.extend(gather_matchup_stats(matchup['matchup'], week))

    return pl.DataFrame(all_stats)


def get_aggregated_stats(df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
    """Aggregates weekly stats into per-week averages based on timeframe.

    Counting stats are averaged per week. GAA and SV% are recomputed from the summed
    underlying stats, so weeks with more goalie starts carry more weight.

    Args:
        df (pl.DataFrame): Raw weekly stats DataFrame
        timeframe (str): 'Last 2 Weeks' or 'Full Season'

    Returns:
        pl.DataFrame: Aggregated stats per team
    """
    if df.is_empty():
        return df

    if timeframe == 'Last 2 Weeks':
        latest_week = int(df.select(pl.col('week').max()).item())
        df = df.filter(pl.col('week') >= latest_week - 1)

    # Back out goalie minutes from GAA = GA * 60 / TOI, so GAA can be weighted properly.
    # Minutes can't be recovered from a 0.00 GAA week, so those weeks drop out of GAA.
    df = df.with_columns(
        pl.when(pl.col('GAA') > 0)
        .then(pl.col('GA') * 60 / pl.col('GAA'))
        .otherwise(None)
        .alias('_TOI')
    )

    counting = ['G', 'A', '+/-', 'PPP', 'SOG', 'HIT', 'W', 'SHO']
    return df.group_by('team').agg(
        [pl.col(c).mean() for c in counting]
        + [
            (pl.col('GA').sum() * 60 / pl.col('_TOI').sum()).alias('GAA'),
            (pl.col('SV').sum() / pl.col('SA').sum()).alias('SV%'),
        ]
    )


def get_my_team_name(league: yfa.League) -> str:
    """Returns the name of the logged-in user's team in the league."""
    return league.teams()[league.team_key()]['name']


def run() -> tuple[pl.DataFrame, str]:
    """Fetches weekly stats for every team in the league, along with my team's name."""
    session = create_session()
    league = yfa.League(session, get_league_id(session))

    return get_league_weekly_stats(league), get_my_team_name(league)


if __name__ == '__main__':
    stats_df, my_team = run()
    print(f"My team: {my_team}")
    print(stats_df)
    print(get_aggregated_stats(stats_df, 'Full Season'))
