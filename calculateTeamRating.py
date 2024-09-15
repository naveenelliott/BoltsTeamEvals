import pandas as pd
import streamlit as st
from scipy.stats import norm

mean_18 = 14.4785
std_18 = 3.1594
mean_forward = 72.7022
std_forward = 3.4743
xg_per_shot_bolts_avg = 0.255
xg_per_shot_bolts_std = 0.06021
xg_per_shot_opp = 0.25
xg_per_shot_opp_std = 0.05
pass_average = 80.6256
pass_std = 2.879
regain_average = 25
regain_std = 4
mean_progr_regain = 74.0805
std_progr_regain = 3.8596
mean_total_pass = 429.0984
std_total_pass = 34.4815
mean_linebreaks = 30.7793
std_linebreaks = 10.9468
mean_progr_att = 63.6586
std_progr_att = 2.2294
mean_ppm = 0.43428
std_ppm = 0.033873
mean_da = 2.8864
std_da = 0.9255

# need to get an actual average and standard deviation here
mean_chance_created = 12.94
std_chance_created = 3

means = {
    'Opp xG per Shot': xg_per_shot_opp,
    'Time Until Regain': regain_average,
    'Progr Regain ': mean_progr_regain,
    'Progr Pass Completion ': pass_average,
    'Total Passes': mean_total_pass,
    'Pass Completion ': pass_average,
    'Passes per Min': mean_ppm,
    'Progr Pass Attempt ': mean_progr_att,
    'Line Break': mean_linebreaks,
    'Pass into Oppo Box': mean_18,
    'xG per Shot': xg_per_shot_bolts_avg,
    'Chance Created': mean_chance_created
}

stds = {
    'Opp xG per Shot': xg_per_shot_opp_std,
    'Time Until Regain': regain_std,
    'Progr Regain ': std_progr_regain,
    'Progr Pass Completion ': pass_std,
    'Total Passes': std_total_pass,
    'Pass Completion ': pass_std,
    'Passes per Min': std_ppm,
    'Progr Pass Attempt ': std_progr_att,
    'Line Break': std_linebreaks,
    'Pass into Oppo Box': std_18,
    'xG per Shot': xg_per_shot_bolts_std,
    'Chance Created': std_chance_created
}


def calculatingTeamRating(team, weekly_report, actions, xg):
    # Filtering based on the selected team
    weekly_report = weekly_report.loc[weekly_report['Team Name'] == team]
    actions = actions.loc[actions['Team'] == team]
    xg = xg.loc[xg['Bolts Team'] == team]

    weekly_report.drop(columns=['Team Name', 'Opposition', 'Match Date', 'Unique Identifier'], inplace=True)

    average_row = weekly_report.mean()

    # Optionally, give this row a name (e.g., 'Average')
    average_row['Team Name'] = 'Average'

    # Create a new DataFrame with just this row
    weekly_report = pd.DataFrame([average_row])


    end = weekly_report.copy()

    end.reset_index(drop=True, inplace=True)
    st.write(end)

    bolts_df = xg[xg['Team'].str.contains('Bolts')]
    opp_df = xg[~xg['Team'].str.contains('Bolts')]

    # Group by the desired columns and aggregate
    bolts_agg = bolts_df.groupby(['Bolts Team', 'Match Date', 'Opposition']).agg(
        Bolts_xG=('xG', 'sum'),
        Bolts_Count=('xG', 'size')
    ).reset_index()

    opp_agg = opp_df.groupby(['Bolts Team', 'Match Date', 'Opposition']).agg(
        Opp_xG=('xG', 'sum'),
        Opp_Count=('xG', 'size')
    ).reset_index()

    overall_xg = pd.merge(bolts_agg, opp_agg, on=['Bolts Team', 'Match Date', 'Opposition'], how='outer')
    overall_xg.rename(columns={'Bolts Team': 'Team'}, inplace=True)
    end['xG per Shot'] = overall_xg['Bolts_xG']/overall_xg['Bolts_Count']
    end['Opp xG per Shot'] = overall_xg['Opp_xG']/overall_xg['Opp_Count']

    _ = """
    time_of_poss_list = []
    time_until_regain_list = []

    # Get unique combinations of 'Team', 'Match Date', and 'Opposition'
    unique_combinations = actions[['Team', 'Match Date', 'Opposition']].drop_duplicates()

    # Iterate over each unique combination
    for _, row in unique_combinations.iterrows():
        team = row['Team']
        match_date = row['Match Date']
        opposition = row['Opposition']
        
        # Filter the DataFrame for the current combination
        filtered_actions = actions[(actions['Team'] == team) & 
                                (actions['Match Date'] == match_date) & 
                                (actions['Opposition'] == opposition)]
        filtered_actions.reset_index(drop=True, inplace=True)
        
        # Run your custom function
        time_of_poss, time_until_regain = formattingFileForRegain(filtered_actions)
        
        # Append results to the lists
        time_of_poss_list.append(time_of_poss)
        time_until_regain_list.append(time_until_regain)

    # Calculate the mean of time until regain
    time_until_regain = np.mean(time_until_regain_list)


    end['Time Until Regain'] = time_until_regain
    """

    #new_order = ['Opp xG per Shot', 'Time Until Regain', 'Progr Regain ',
    #'Progr Pass Completion ', 'Total Passes', 'Pass Completion ', 'Passes per Min', 'Progr Pass Attempt ', 
    #'Line Break', 'Pass into Oppo Box', 'xG per Shot', 'Chance Created']
    new_order = ['Opp xG per Shot', 'Progr Regain ',
    'Progr Pass Completion ', 'Total Passes', 'Pass Completion ', 'Passes per Min', 'Progr Pass Attempt ', 
    'Line Break', 'Pass into Oppo Box', 'xG per Shot', 'Chance Created']
    end = end[new_order]

    # Apply the conversion to percentiles
    for col in end.columns:
        if col in means and col in stds:
            mean = means[col]
            std = stds[col]
            end[col] = norm.cdf(end[col], loc=mean, scale=std) * 100
        else:
            # General percentile rank if no specific distribution is known
            end[col] = end[col].rank(pct=True) * 100


    end['Opp xG per Shot'] = 100 - end['Opp xG per Shot']
    #end['Time Until Regain'] = 100 - end['Time Until Regain']

    # parameter list
    #params = [
    #    'Opp xG per Shot', 'Time Until Regain', 'Progr Regain ',
    #    'Progr Pass Completion ', 'Total Passes', 'Pass Completion ', 'Passes per Min', 'Progr Pass Attempt ', 
    #    'Line Break', 'Pass into Oppo Box', 'xG per Shot', 'Chance Created'
    #]
    params = [
        'Opp xG per Shot', 'Progr Regain ',
        'Progr Pass Completion ', 'Total Passes', 'Pass Completion ', 'Passes per Min', 'Progr Pass Attempt ', 
        'Line Break', 'Pass into Oppo Box', 'xG per Shot', 'Chance Created'
    ]

    # value list
    values = [int(end[col]) for col in params]

    return values
