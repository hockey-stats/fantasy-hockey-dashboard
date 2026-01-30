import os
import logging
from datetime import datetime, timedelta

import polars as pl
import pyhockey as ph
import yahoo_fantasy_api as yfa
from yahoo_oauth import OAuth2
from unidecode import unidecode


oauth_logger = logging.getLogger('yahoo_oauth')
oauth_logger.disabled = True


## Constants #########################################################################
LEAGUE_NAME = 'Slush puppie Invitational'

TEAM_MAPPING = {
    'SJ': 'SJS',
    'LA': 'LAK',
    'TB': 'TBL',
    'NJ': 'NJD',
}
## End Constants #####################################################################


def standardize_team_names(team: str) -> str:
    """
    Function used to adjust team abbreviations provided by the Yahoo API to match
    those used by pybaseball.

    :param str team: The team name abbreviation provided by the Yahoo API
    :return str: Team name abbreviation used by pybaseball
    """
    return TEAM_MAPPING.get(team, team)


def create_session() -> OAuth2:
    """ Creates the OAuth2 session from a local json. Refreshes token if necessary.

    :return OAuth2: Active OAuth2 session
    """
    if not os.path.isfile('oauth.json'):
        with open('oauth.json', 'w', encoding='utf-8') as f:
            f.write(os.environ['OAUTH_TOKEN'])
    sc = OAuth2(None, None, from_file='oauth.json')
    return sc


def collect_goalie_stats(player_ids: list[str], league: yfa.League) -> pl.DataFrame:
    """ Uses a list of player IDs to pull their stats and organize into a DF.

    Args:
        player_ids (list[str]): List of player IDs, as strings, representing goalies
        league (yfa.League): The League object to query against

    Returns:
        pl.DataFrame: DF containing data for each goalie
    """
    player_details = league.player_details(player_ids)

    player_stats_week = league.player_stats(player_ids, 'lastweek')
    player_stats_month = league.player_stats(player_ids, 'lastmonth')
    player_stats_season = league.player_stats(player_ids, 'season')

    p_dict = {
        'name': [],
        'id': [],
        'team': [],
        'positions': [],
        'w': [],
        'ga': [],
        'gaa': [],
        'sv': [],
        'sa': [],
        'sv%': [],
        'sho': [],
        'term': [],
    }

    for player_stats, term in zip([player_stats_week, player_stats_month, player_stats_season],
                                  ['week', 'month', 'season']):
        for details, stats in zip(player_details, player_stats):

            if stats['W'] == '-' or stats['GAA'] == '-' or stats['SV%'] == '-':
                continue

            p_dict['name'].append(unidecode(details['name']['full']))
            p_dict['id'].append(stats['player_id'])
            p_dict['team'].append(standardize_team_names(details['editorial_team_abbr']))
            p_dict['positions'].append(','.join([pos['position'] \
                                                for pos in details['eligible_positions']]))
            p_dict['w'].append(int(stats['W']))
            p_dict['ga'].append(int(stats['GA']))
            p_dict['gaa'].append(float(stats['GAA']))
            p_dict['sv'].append(int(stats['SV']))
            p_dict['sa'].append(int(stats['SA']))
            p_dict['sv%'].append(float(stats['SV%']))
            p_dict['sho'].append(int(stats['SHO']))
            p_dict['term'].append(term)

    y_df = pl.DataFrame(p_dict)

    # Get advanced data from pyhockey to augment yahoo data
    ph_df = pl.concat([get_advanced_goalie_data('week'),
                       get_advanced_goalie_data('month'),
                       get_advanced_goalie_data('season')])

    df = y_df.join(ph_df, how='left', on=['name', 'team', 'term'])

    return df


def collect_skater_stats(player_ids: list[str], league: yfa.League) -> pl.DataFrame:
    """ Uses a list of player IDs to pull their stats and organize into a DF.

    Args:
        player_ids (list[str]): List of player IDs, as strings, representing skaters
        league (yfa.League): The League object to query against

    Returns:
        pl.DataFrame: DF containing data for each skater
    """
    player_details = league.player_details(player_ids)

    player_stats_week = league.player_stats(player_ids, 'lastweek')
    player_stats_month = league.player_stats(player_ids, 'lastmonth')
    player_stats_season = league.player_stats(player_ids, 'season')

    p_dict = {
        'name': [],
        'id': [],
        'team': [],
        'positions': [],
        'g': [],
        'a': [],
        'pm': [],
        'ppp': [],
        'sog': [],
        'h': [],
        'term': [],
    }

    for player_stats, term in zip([player_stats_week, player_stats_month, player_stats_season],
                                  ['week', 'month', 'season']):
        for details, stats in zip(player_details, player_stats):

            if stats['G'] == '-':
                continue

            p_dict['name'].append(unidecode(details['name']['full']))
            p_dict['id'].append(stats['player_id'])
            p_dict['team'].append(standardize_team_names(details['editorial_team_abbr']))
            p_dict['positions'].append(','.join([pos['position'] \
                                                for pos in details['eligible_positions']]))
            p_dict['g'].append(int(stats['G']))
            p_dict['a'].append(int(stats['A']))
            p_dict['pm'].append(int(stats['+/-']))
            p_dict['ppp'].append(int(stats['PPP']))
            p_dict['sog'].append(int(stats['SOG']))
            p_dict['h'].append(int(stats['HIT']))
            p_dict['term'].append(term)

    y_df = pl.DataFrame(p_dict)

    # Get DF of advanced data from pyhockey to join in
    ph_df = pl.concat([get_advanced_skater_data('week'),
                       get_advanced_skater_data('month'),
                       get_advanced_skater_data('season')])

    df = y_df.join(ph_df, how='left', on=['name', 'team', 'term'])

    return df


def get_advanced_skater_data(term: str) -> pl.DataFrame:
    """ Returns a DF of additional skater data from pyhockey

    Args:
        term (str): One of 'week', 'month', or 'season', the term over which to gather
                    game data.

    Raises:
        ValueError: If an invalid term input is provided.

    Returns:
        pl.DataFrame: DF containing additional data to augment data provided by yahoo API.
    """

    today = datetime.now()
    if term == 'season':
        start_date = None
    elif term == 'month':
        start_date = datetime.strftime(today - timedelta(days=31), '%Y-%m-%d')
    elif term == 'week':
        start_date = datetime.strftime(today - timedelta(days=7), '%Y-%m-%d')
    else:
        raise ValueError('Incorrect term input provided, must be one of [season, month, week]')

    df = ph.skater_games(season=today.year if today.month >= 10 else today.year - 1,
                         start_date=start_date, quiet=True)

    all_sit_df = df.filter(pl.col('situation') == 'all')\
        .group_by(['name', 'position', 'team'])\
        .agg(pl.col('individualxGoals').sum())

    icetime_df = df.pivot('situation', index=['name', 'team', 'position', 'gameID'],
                          values='iceTime')\
        .group_by(['name', 'team', 'position'])\
        .agg(pl.col('ev').mean(), pl.col('pp').mean())

    output = all_sit_df.join(icetime_df, how='inner', on=['name', 'team', 'position'])

    output = output.with_columns(
        pl.lit(term).alias('term')
    )

    output = output.drop("position")

    output = output.rename({
        'individualxGoals': 'ixG',
        'ev': 'EV ToI/g',
        'pp': 'PP ToI/g'
    })

    return output


def get_advanced_goalie_data(term: str) -> pl.DataFrame:
    """ Returns a DF of additional goalie data from pyhockey

    Args:
        term (str): One of 'week', 'month', or 'season', the term over which to gather
                    game data.

    Raises:
        ValueError: If an invalid term input is provided.

    Returns:
        pl.DataFrame: DF containing additional data to augment data provided by yahoo API.
    """

    today = datetime.now()
    if term == 'season':
        start_date = None
    elif term == 'month':
        start_date = datetime.strftime(today - timedelta(days=31), '%Y-%m-%d')
    elif term == 'week':
        start_date = datetime.strftime(today - timedelta(days=7), '%Y-%m-%d')
    else:
        raise ValueError('Incorrect term input provided, must be one of [season, month, week]')

    df = ph.goalie_games(season=today.year if today.month >= 10 else today.year - 1,
                         start_date=start_date, situation='all', quiet=True)

    df = df.with_columns(
        (pl.col('xGoalsAgainst') - pl.col('goalsAgainst')).alias('GSAX')
    )

    df = df.group_by(['name', 'team'])\
        .agg(pl.col('GSAX').mean(), pl.col('xGoalsAgainst').mean(), pl.col('gameID').count())

    output = df.rename({
        'GSAX': 'GSAX/g',
        'xGoalsAgainst': 'xGA/g',
        'gameID': 'GP'
    })

    output = output.with_columns(
        pl.lit(term).alias('term')
    )

    return output


def compute_z_scores(df: pl.DataFrame, player_type: str) -> pl.DataFrame:
    """ Computes z-scores for each column in the DataFrame and returns one with an extra column
    for the average of these z-scores for each term.

    Args:
        df (pl.DataFrame): Raw stats for each player
        player_type (str): Either 'skater' or 'goalie'

    Returns:
        pl.DataFrame: Input DF with z-scores added in
    """
    # Dict to store DataFrames for each term
    dfs = dict()

    if player_type == 'skaters':
        columns = ['g', 'a', 'pm', 'ppp', 'sog', 'h']
    else:
        columns = ['w', 'gaa', 'sv%', 'sho']

    for term in ['season', 'month', 'week']:
        # Compute z-scores seperately for each term
        x = df.filter(pl.col('term') == term)

        for col in columns:
            mult = -1 if col == 'gaa' else 1
            x = x.with_columns(
                (((pl.col(col) - pl.mean(col)) / pl.std(col)) * mult).alias(f"z_{col}")
            )

        if player_type == 'skaters':
            x = x.with_columns(
                ((pl.col('z_g') +\
                  pl.col('z_a') +\
                  pl.col('z_pm') +\
                  pl.col('z_ppp') +\
                  pl.col('z_sog') +\
                  pl.col('z_h')) / 6).alias('z_total')
            )
        else:
            x = x.with_columns(
                ((pl.col('z_w') +\
                  pl.col('z_gaa') +\
                  pl.col('z_sv%') +\
                  pl.col('z_sho')) / 4).alias('z_total')
            )

        # Add a column for ranking by z_total
        x = x.with_columns(pl.struct('z_total').rank('ordinal', descending=True).alias("Rank"))

        # Drop columns we don't want in final output (i.e. the z_ columns)
        drop_columns = [f'z_{col}' for col in columns] + ['z_total']
        x = x.drop(drop_columns)

        # Save to dict
        dfs[term] = x.clone()

    # Return the final output as a concatenation of all the term-specific DFs
    final = pl.concat(list(dfs.values()))

    return final


def get_players_from_own_team(position: str, league: yfa.League, session: OAuth2) -> pl.DataFrame:
    """
    Pulls from the fantasy API all players of the given position from my own team.

    Args:
        position (str): Position for which to pull players.
        league (yfa.League): The League object from the yfa library, representing our league.
        session (OAuth2): OAuth2 session used for authentication.

    Returns:
        pl.DataFrame: DataFrame containing the players and their stats.
    """
    # Get all teams and find one owned by the caller of the API
    teams = league.teams()
    for team in teams:
        if teams[team].get('is_owned_by_current_login', False):
            my_team = yfa.Team(session, team)
            break
    else:
        print("Own team not found, exiting...")
        raise ValueError

    # Collect only the players of the given position
    players = []
    for player in my_team.roster():
        # Ignore players on the IL
        if 'IL' in player['status']:
            continue
        if player['position_type'] == position:
            players.append(player['player_id'])

    if position == 'P':
        df = collect_skater_stats(players, league)
    else:
        df = collect_goalie_stats(players, league)

    return df


def filter_taken(df: pl.DataFrame, free_agents: list[int], my_team: list[int]) -> pl.DataFrame:
    """
    Removes from a player DF every player that's on a team which is not my team, and then adds 
    a column for boolean values corresponding to whether or not a player is on my team.

    Uses player_ids returned by the Yahoo Fantasy API for filtering.

    Args:
        df (pl.DataFrame): DataFrame containing stats for all players
        free_agents (list[int]): Set of IDs of players which are free agents
        my_team list[int]: Set of IDs of players on my team.

    Returns
        pl.DataFrame: DataFrame with taken players removed and 'on_team' column added.
    """

    # Filter out players that are already taken
    df = df.filter(pl.col('id').is_in(free_agents + my_team))

    # Also filter out any injured players
    df = df.filter(pl.col('positions').str.contains('IL').not_())

    # Set 'on_team' to True if player is on my team, False otherwise
    df = df.with_columns(
        pl.when(pl.col('id').is_in(my_team))
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias('on_team')
    )

    return df


def get_league_id(sc: OAuth2) -> str:
    """Gets the ID of the desired fantasy league.

    Args:
        sc (OAuth2): The oauth session used to authenticate.

    Returns:
        str: The ID of the desired league.
    """
    game = yfa.Game(sc, 'nhl')
    my_league = None
    # Choose the league with the correct name
    for league_id in game.league_ids():
        if yfa.League(sc, league_id).__dict__['settings_cache']['name'] == LEAGUE_NAME:
            my_league = league_id

    return league_id



def main() -> None:
    """ Pulls data from Yahoo API and pyhockey to generate CSVs to be used by dashboard. """

    session = create_session()

    league_id = get_league_id(session)
    league = yfa.League(session, league_id)

    # Collect IDs for taken players
    taken = league.taken_players()
    skater_ids = [p['player_id'] for p in taken if p['position_type'] == 'P']
    goalie_ids = [p['player_id'] for p in taken if p['position_type'] == 'G']

    # Add IDs for free agents
    skater_ids.extend(p['player_id'] for p in league.free_agents('P'))
    goalie_ids.extend(p['player_id'] for p in league.free_agents('G'))

    # Organize information into dataframes
    s_df = collect_skater_stats(skater_ids, league)
    g_df = collect_goalie_stats(goalie_ids, league)

    # Get z-scores
    s_df = compute_z_scores(s_df, player_type='skaters')
    g_df = compute_z_scores(g_df, player_type='goalies')

    # Compile list of all free agents to be included in final output
    skater_fas = [p['player_id'] for p in league.free_agents('P')]
    goalie_fas = [p['player_id'] for p in league.free_agents('G')]

    # Do the same with players from my team
    my_team = list(get_players_from_own_team(position='P',
                                             league=league,
                                             session=session)['id']) + \
              list(get_players_from_own_team(position='G',
                                             league=league,
                                             session=session)['id'])

    skater_df = filter_taken(s_df, skater_fas, my_team)
    goalie_df = filter_taken(g_df, goalie_fas, my_team)

    # Rename columns to be more presentable
    skater_df = skater_df.rename({
        'name': 'Name',
        'team': 'Team',
        'positions': 'Position(s)',
        'g': 'G',
        'a': 'A',
        'pm': '+/-',
        'ppp': 'PPP',
        'sog': 'SOG',
        'h': 'HITS'
    })
    skater_df.drop('id')

    goalie_df = goalie_df.rename({
        'name': 'Name',
        'team': 'Team',
        'positions': 'Position(s)',
        'w': 'W',
        'ga': 'GA',
        'gaa': 'GAA',
        'sv': 'SV',
        'sv%': 'SV%',
        'sho': 'SHO'
    })
    goalie_df.drop('id')

    # Save output
    skater_df.write_csv('skater_data.csv')
    goalie_df.write_csv('goalie_data.csv')


if __name__ == '__main__':
    main()
