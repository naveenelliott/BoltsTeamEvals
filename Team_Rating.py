import pandas as pd
import streamlit as st
import glob
import os
from mplsoccer import PyPizza
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.font_manager import FontProperties
from GettingTimeUntilRegain import formattingFileForRegain
from xGModel import xGModel
import base64
import numpy as np

st.set_page_config(
    page_title="Bolts Team Rating App",
    page_icon='soccer-coach-clipart-xl.png',
    layout="centered",
)

# Inject custom CSS
st.markdown(
    """
    <style>
    /* Background color for the whole app */
    .stApp {
        background-color: #faf0e6;
    }

    /* Style the Markdown elements */
    .stMarkdown h1, h2, h3, h4, h5, h6 {
        color: #33475b;  /* Change header colors */
    }

    .stMarkdown p {
        color: #505a73;  /* Change paragraph text color */
    }

    /* Style buttons */
    .stButton > button {
        background-color: #6bb2e2;  /* Button background */
        color: white;  /* Button text */
        border: 2px solid black;  /* Black border around buttons */
        border-radius: 5px;  /* Optional: rounded corners */
        padding: 10px 20px;  /* Optional: padding inside the button */
        font-size: 16px;  /* Optional: font size */
        cursor: pointer;  /* Ensure cursor indicates a clickable button */
    }
    </style>
    """,
    unsafe_allow_html=True
)


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

# need to get an actual average and standard deviation here
mean_chance_created = 12.94
std_chance_created = 3

font_path = 'Roboto-Regular.ttf'
font_normal = FontProperties(fname=font_path)
font_path = 'Roboto-Italic.ttf'
font_italic = FontProperties(fname=font_path)
font_path = 'Roboto-Bold.ttf'
font_bold = FontProperties(fname=font_path)

font_path = 'CookbookNormalRegular-6YmjD.ttf'
cook = FontProperties(fname=font_path)

image_path = 'Boston_Bolts.png'  # Replace with the actual path to your image
image = plt.imread(image_path)

font_path = 'WorkSans-Bold.ttf'
work_sans = FontProperties(fname=font_path)

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

folder_path = 'Actions PSD'

# Find all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# List to hold individual DataFrames
df_list = []

# Loop through the CSV files and read them into DataFrames
for file in csv_files:
    df = pd.read_csv(file)
    df.columns = df.loc[4]
    df = df.loc[5:].reset_index(drop=True)
    df['Match Date'] = pd.to_datetime(df['Match Date']).dt.strftime('%m/%d/%Y')
    df_list.append(df)

# Concatenate all DataFrames into a single DataFrame
actions = pd.concat(df_list, ignore_index=True)
actions.loc[actions['Opposition'] == 'St Louis', 'Match Date'] = '12/09/2023'



actions['Unique Identifier'] = actions['Team'] + ' ' + actions['Opposition'] + ' ' + actions['Match Date']

folder_path = 'WeeklyReport PSD'

# Find all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# List to hold individual DataFrames
df_list = []

# Loop through the CSV files and read them into DataFrames
for file in csv_files:
    df = pd.read_csv(file)
    df.columns = df.iloc[3]
    df = df.iloc[4:]
    df = df.reset_index(drop=True)
    start_index = df.index[df["Period Name"] == "Round By Team"][0]

    # Find the index where "period name" is equal to "running by position player"
    end_index = df.index[df["Period Name"] == "Running By Player"][0]

    df = df.iloc[start_index:end_index]
    cols_we_want = [
    'Pass into Oppo Box', 'Progr Regain ',
    'Progr Pass Completion ', 'Success', 'Unsuccess', 'Pass Completion ', 'Progr Pass Attempt ', 
    'Line Break', 'mins played', 'Chance Created', 'Team Name', 'Opposition', 'Match Date', 'Opp Effort on Goal'
    ]
    float_columns = [
        'Pass into Oppo Box', 'Progr Regain ',
    'Progr Pass Completion ', 'Success', 'Unsuccess', 'Pass Completion ', 'Progr Pass Attempt ', 
    'Line Break', 'mins played', 'Chance Created',  'Opp Effort on Goal'
    ]

    df = df[cols_we_want]

    df[float_columns] = df[float_columns].astype(float)

    df['Total Passes'] = df['Success'] + df['Unsuccess']

    df['Passes per Min'] = df['Total Passes']/df['mins played']

    per90 = ['Pass into Oppo Box', 'Total Passes', 'Line Break', 'Chance Created']

    for col in per90:
        df[col] = df.apply(lambda row: (row[col] / row['mins played'] * 990) if row['mins played'] > 0 else 0, axis=1)

    df.drop(columns=['Success', 'Unsuccess', 'mins played'], inplace=True)
    df['Match Date'] = pd.to_datetime(df['Match Date']).dt.strftime('%m/%d/%Y')
    df_list.append(df)

# Concatenate all DataFrames into a single DataFrame
weekly_report = pd.concat(df_list, ignore_index=True)
temp_shots = weekly_report['Opp Effort on Goal'].sum()
weekly_report.loc[weekly_report['Opposition'] == 'St Louis', 'Match Date'] = '12/09/2023'

weekly_report['Unique Identifier'] = weekly_report['Team Name'] + ' ' + weekly_report['Opposition'] + ' ' + weekly_report['Match Date']

weekly_report = weekly_report[weekly_report['Unique Identifier'].isin(actions['Unique Identifier'])]


folder_path = 'xG Input Files'

# Find all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# List to hold individual DataFrames
df_list = []

# Loop through the CSV files and read them into DataFrames
for file in csv_files:
    df = pd.read_csv(file)
    df_list.append(df)

# Concatenate all DataFrames into a single DataFrame
fc_python = pd.concat(df_list, ignore_index=True)

# Making sure everything is aligned on one side
def flip_coordinates(x, y):
    # Flip x and y coordinates horizontally
    flipped_x = field_dim - x
    flipped_y = field_dim - y  # y remains unchanged in this transformation
    
    return flipped_x, flipped_y

field_dim = 100
# Iterating through coordinates and making them on one side
flipped_points = []
for index, row in fc_python.iterrows():
    if row['X'] < 50:
        flipped_x, flipped_y = flip_coordinates(row['X'], row['Y'])
        fc_python.at[index, 'X'] = flipped_x
        fc_python.at[index, 'Y'] = flipped_y

# these are the ideal columns
cols = ['Player Full Name', 'Team', 'Match Date', 'Opposition', 'Action', 'Time', 'Video Link']
xg_actions = actions[cols]

# these are the shots we want
wanted_actions = ['Att Shot Blockd', 'Blocked Shot', 'Goal', 'Goal Against', 'Header on Target', 
                  'Header off Target', 'Opp Effort on Goal', 'Save Held', 'Save Parried', 'Shot off Target', 
                  'Shot on Target']
xg_actions = xg_actions.loc[xg_actions['Action'].isin(wanted_actions)].reset_index(drop=True)
# renaming for the join
xg_actions.rename(columns={'Team': 'Bolts Team'}, inplace=True)

# if the opponent shots are 0, we factor in blocked shots
# if they aren't 0, then we don't factor in blocked shots
if temp_shots != 0:
    xg_actions = xg_actions.loc[xg_actions['Action'] != 'Blocked Shot'].reset_index(drop=True)

# this is handeling duplicated PlayerStatData shots 
temp_df = pd.DataFrame(columns=xg_actions.columns)
prime_actions = ['Opp Effort on Goal', 'Shot on Target']
remove_indexes = []
for index in range(len(xg_actions) - 1):
    if xg_actions.loc[index, 'Time'] == xg_actions.loc[index+1, 'Time']:
        temp_df = pd.concat([temp_df, xg_actions.loc[[index]], xg_actions.loc[[index + 1]]], ignore_index=False)
        bye1 = temp_df.loc[temp_df['Action'].isin(prime_actions)]
        # these are the indexes we want to remove
        remove_indexes.extend(bye1.index)
        
    temp_df = pd.DataFrame(columns=xg_actions.columns)     


fc_python['Match Date'] = pd.to_datetime(fc_python['Match Date']).dt.strftime('%m/%d/%Y')

# combining into xG dataframe we want
combined_xg = pd.merge(fc_python, xg_actions, on=['Bolts Team', 'Match Date', 'Time'], how='inner')


# running the model on our dataframe
xg = xGModel(combined_xg)

# Convert the image to base64
logo_image = "Boston_Bolts.png"

# Inject CSS and HTML for title and logo
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Convert the image to base64
image_base64 = get_image_base64("Boston_Bolts.png")

# Inject CSS and HTML for title and logo
st.markdown(
    f"""
    <style>
        .centered-title-logo {{
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .centered-title-logo h1 {{
            margin: 0;
            font-size: 40px;
        }}
        .centered-title-logo img {{
            width: 100px;  /* Adjust the width of the image */
            height: 80px;  /* Maintain the aspect ratio */
        }}
    </style>
    <div class="centered-title-logo">
        <h1>Boston Bolts Playing Style App</h1>
        <img src="data:image/png;base64,{image_base64}" alt="Boston Bolts Logo">
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("Select the Team, Opponent, and Date to view the team ranking. Team percentiles are compared with all teams in the club")




# Selecting the Bolts team
teams = list(sort(weekly_report['Team Name'].unique()))
selected_team = st.session_state.get('selected_team', teams[0])
if selected_team not in teams:
    selected_team = teams[0]  # Default to the first date if not found

selected_team = st.selectbox('Choose the Team:', teams, index=teams.index(selected_team))
st.session_state['selected_team'] = selected_team

# Filtering based on the selected team
weekly_report = weekly_report.loc[weekly_report['Team Name'] == st.session_state['selected_team']]
actions = actions.loc[actions['Team'] == st.session_state['selected_team']]
xg = xg.loc[xg['Bolts Team'] == st.session_state['selected_team']]

weekly_report.drop(columns=['Team Name', 'Opposition', 'Match Date', 'Unique Identifier'], inplace=True)

average_row = weekly_report.mean()

# Optionally, give this row a name (e.g., 'Average')
average_row['Team Name'] = 'Average'

# Create a new DataFrame with just this row
weekly_report = pd.DataFrame([average_row])


end = weekly_report.copy()

end.reset_index(drop=True, inplace=True)

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


new_order = ['Opp xG per Shot', 'Time Until Regain', 'Progr Regain ',
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
end['Time Until Regain'] = 100 - end['Time Until Regain']

# parameter list
params = [
    'Opp xG per Shot', 'Time Until Regain', 'Progr Regain ',
    'Progr Pass Completion ', 'Total Passes', 'Pass Completion ', 'Passes per Min', 'Progr Pass Attempt ', 
    'Line Break', 'Pass into Oppo Box', 'xG per Shot', 'Chance Created'
]

# value list
values = [int(end[col]) for col in params]

# color for the slices and text
slice_colors = ['gray'] * 3 + ["#6bb2e2"] * 3 + ["#263a6a"] * 3 + ['black'] * 3 
text_colors =  ["#F2F2F2"] * 3 + ["#000000"] * 3 + ["#F2F2F2"] * 3 + ['#F2F2F2'] * 3

# instantiate PyPizza class
baker = PyPizza(
    params=params,                  # list of parameters
    background_color="#EBEBE9",     # background color
    straight_line_color="#EBEBE9",  # color for straight lines
    straight_line_lw=1,             # linewidth for straight lines
    last_circle_lw=0,               # linewidth of last circle
    other_circle_lw=0,              # linewidth for other circles
    inner_circle_size=10            # size of inner circle
)

# plot pizza
fig, ax = baker.make_pizza(
    values,                          # list of values
    figsize=(8, 8.5),                # adjust figsize according to your need
    color_blank_space="same",        # use same color to fill blank space
    slice_colors=slice_colors,       # color for individual slices
    value_colors=text_colors,        # color for the value-text
    value_bck_colors=slice_colors,   # color for the blank spaces
    blank_alpha=0.4,                 # alpha for blank-space colors
    kwargs_slices=dict(
        edgecolor="#F2F2F2", zorder=2, linewidth=1
    ),                               # values to be used when plotting slices
    kwargs_params=dict(
        color="#000000", fontsize=12.5,
        fontproperties=font_normal, va="center"
    ),                               # values to be used when adding parameter
    kwargs_values=dict(
        color="#000000", fontsize=11,
        fontproperties=font_normal, zorder=3,
        bbox=dict(
            edgecolor="#000000", facecolor="cornflowerblue",
            boxstyle="round,pad=0.2", lw=1
        )
    )                                # values to be used when adding parameter-values
)

for i, (text, color) in enumerate(zip(ax.texts[:len(params)], slice_colors)):
    text.set_color(color)



fig.text(
    0.85, 0.85,
    "Defending",
    size=16,
    ha="center", fontproperties=font_bold, color="gray"
)

fig.text(
    0.875, 0.15,
    "Possession",
    size=16,
    ha="center", fontproperties=font_bold, color="#6bb2e2"
)

fig.text(
    0.175, 0.15,
    "Progression",
    size=16,
    ha="center", fontproperties=font_bold, color="#263a6a"
)


fig.text(
    0.175, 0.85,
    "Attacking",
    size=16,
    ha="center", fontproperties=font_bold, color="black"
)


fig.set_facecolor('#faf0e6')
plt.gca().set_facecolor('#faf0e6')

fig.set_dpi(600)

st.pyplot(fig)
