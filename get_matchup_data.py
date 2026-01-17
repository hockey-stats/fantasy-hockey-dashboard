import logging

import polars as pl
import yahoo_fantasy_api as yfa

from update_data import create_session, get_league_id


STAT_MAP = {
    '1': 'G',
    '2': 'A',
    '4': '+/-',
    '8': 'PPP',
    '14': 'SOG',
    '31': 'HIT',
    '19': 'W',
    '23': 'GAA',
    '26': 'SV%',
    '27': 'SHO'
}

def gather_stats(matchup: dict[str: dict]) -> pl.DataFrame:
    """Parses matchup data for the stats for each team, returns as DataFrame.

    Args:
        matchup (dict): Dict object containing info about teams in the matchup

    Returns:
        pl.DataFrame: Stats for each team.
    """
    data_dict = {
        'team': [],
        'G': [],
        'A': [],
        '+/-': [],
        'PPP': [],
        'SOG': [],
        'HIT': [],
        'W': [],
        'GAA': [],
        'SV%': [],
        'SHO': [],
        'GR': [],
    }

    for i in ['0', '1']:
        team_info = matchup['0']['teams'][i]['team'][0]
        for entry in team_info:
            if type(entry) == dict and entry.get('name', False):
                data_dict['team'].append(entry['name'])
                break
        stats = matchup['0']['teams'][i]['team'][1]
        for entry in stats['team_stats']['stats']:
            if stat_name := STAT_MAP.get(entry['stat']['stat_id']):
                stat_value = entry['stat']['value']
                data_dict[stat_name].append(stat_value)

        games_remaining = stats['team_remaining_games']['total']['remaining_games']
        data_dict['GR'].append(games_remaining)

    df = pl.DataFrame(data_dict)
    return df


def get_my_matchup(league: yfa.League) -> dict[str: dict]:
    """Get details for my current matchup."""
    my_team_id = league.team_key()

    all_matchups = league.matchups()['fantasy_content']['league'][1]['scoreboard']['0']['matchups']
    for matchup in all_matchups.values():
        teams = matchup['matchup']['0']['teams']
        a = teams['0']['team'][0]
        b = teams['1']['team'][0]

        for entry in a + b:
            if type(entry) == dict and entry.get('team_key', None) == my_team_id:
                my_matchup = matchup
                # If my matchup is found, break this loop and consequently the outer loop
                break 
        else:
            continue
        break

    return my_matchup['matchup']


def main() -> pl.DataFrame:
    """Uses API data to fetch details for this week's matchup."""

    session = create_session()
    league_id = get_league_id(session)
    league = yfa.League(session, league_id)

    matchup = get_my_matchup(league)

    df = gather_stats(matchup)
    return df


if __name__ == '__main__':
    main()