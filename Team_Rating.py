import pandas as pd
import streamlit as st
import glob
import os
from mplsoccer import Radar
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.font_manager import FontProperties
from GettingTimeUntilRegain import formattingFileForRegain
from xGModel import xGModel
import base64
import numpy as np
from calculateTeamRating import calculatingTeamRating

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
    'Pass into Oppo Box', 'Progr Regain ', 'Progr Rec', 'Unprogr Rec', 'Progr Inter', 'Unprogr Inter', 
    'Stand. Tackle', 'Unsucc Stand. Tackle', 'Tackle', 'Unsucc Tackle',
    'Progr Pass Completion ', 'Success', 'Unsuccess', 'Pass Completion ', 'Progr Pass Attempt ', 
    'Line Break', 'mins played', 'Chance Created', 'Team Name', 'Opposition', 'Match Date', 'Opp Effort on Goal'
    ]
    float_columns = [
    'Pass into Oppo Box', 'Progr Regain ', 'Progr Inter', 'Unprogr Inter', 'Progr Rec', 'Unprogr Rec', 
    'Stand. Tackle', 'Unsucc Stand. Tackle', 'Tackle', 'Unsucc Tackle',
    'Progr Pass Completion ', 'Success', 'Unsuccess', 'Pass Completion ', 'Progr Pass Attempt ', 
    'Line Break', 'mins played', 'Chance Created',  'Opp Effort on Goal'
    ]

    df = df[cols_we_want]

    df[float_columns] = df[float_columns].astype(float)

    df['TDA'] = df['Progr Rec'] + df['Unprogr Rec'] + df['Progr Inter'] + df['Unprogr Inter'] + df['Tackle'] + df['Unsucc Tackle'] + df['Stand. Tackle'] + df['Unsucc Stand. Tackle']

    df['Total Passes'] = df['Success'] + df['Unsuccess']

    df['Passes per Min'] = df['Total Passes']/df['mins played']

    per90 = ['Pass into Oppo Box', 'Total Passes', 'Line Break', 'Chance Created', 'TDA']

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
st.markdown("Select the Team to view the team ranking. Team percentiles are compared with all teams in the club")


our_columns = st.columns(3)

with our_columns[0]:
    # Selecting the Bolts team
    weekly_report = weekly_report.dropna(subset=['Team Name']).reset_index(drop=True)
    teams = sorted(list(weekly_report['Team Name'].unique()))
    selected_team = st.session_state.get('selected_team', teams[0])
    if selected_team not in teams:
        selected_team = teams[0]  # Default to the first date if not found

    selected_team = st.selectbox('Choose the Team (Black):', teams, index=teams.index(selected_team))
    team = selected_team

values = calculatingTeamRating(team, weekly_report, actions, xg)

with our_columns[1]:
    # Add a 'None' option to the list of teams
    if selected_team in teams:
        team_options = ['None'] + [team for team in teams if team != selected_team]
    else:
        team_options = ['None'] + teams
    # Create a selectbox with the updated options
    selected_team_2 = st.selectbox('Choose the 1st Comp Team (White):', team_options)

if selected_team_2 != 'None':
  other_values = calculatingTeamRating(selected_team_2, weekly_report, actions, xg)

with our_columns[2]:
    team_options_2 = [team for team in teams if team != selected_team]
    # Ensure selected_team_2 is not in the list of options
    team_options_2 = [team for team in team_options_2 if team != selected_team_2]
    team_options_2 = ['None'] + team_options_2
    # Create a selectbox with the updated options
    selected_team_3 = st.selectbox('Choose the 2nd Comp Team (Blue):', team_options_2)

if selected_team_3 != 'None':
  final_values = calculatingTeamRating(selected_team_3, weekly_report, actions, xg)

params = [
    'Opp xG per Shot', 'Progr Regain ',
    'Progr Pass Completion ', 'Total Passes', 'Pass Completion ', 'Passes per Min', 'Progr Pass Attempt ', 
    'Line Break', 'Pass into Oppo Box', 'xG per Shot', 'Chance Created'
]


low = [0] * len(params)
high = [100] * len(params)


radar = Radar(params, low, high,
              # whether to round any of the labels to integers instead of decimal places
              round_int=[True]*len(params),
              num_rings=4,  # the number of concentric circles (excluding center circle)
              # if the ring_width is more than the center_circle_radius then
              # the center circle radius will be wider than the width of the concentric circles
              ring_width=1, center_circle_radius=1)

fig, ax = radar.setup_axis()
rings_inner = radar.draw_circles(ax=ax, facecolor='lightgray', edgecolor='black')

radar1, vertices1 = radar.draw_radar_solid(values, ax=ax,
                                           kwargs={'facecolor': 'black',
                                                   'alpha': 0.6,
                                                   'edgecolor': 'black',
                                                   'lw': 2})

ax.scatter(vertices1[:, 0], vertices1[:, 1],
           c='black', edgecolors='black', marker='o', s=100, zorder=2)

if selected_team_2 != 'None':
    radar1, vertices1 = radar.draw_radar_solid(other_values, ax=ax,
                                           kwargs={'facecolor': 'white',
                                                   'alpha': 0.7,
                                                   'edgecolor': 'black',
                                                   'lw': 2})

    ax.scatter(vertices1[:, 0], vertices1[:, 1],
           c='white', edgecolors='black', marker='o', s=100, zorder=2)
    
if selected_team_3 != 'None':
    radar1, vertices1 = radar.draw_radar_solid(final_values, ax=ax,
                                           kwargs={'facecolor': 'lightblue',
                                                   'alpha': 0.7,
                                                   'edgecolor': 'black',
                                                   'lw': 2})

    ax.scatter(vertices1[:, 0], vertices1[:, 1],
           c='lightblue', edgecolors='black', marker='o', s=100, zorder=2)

range_labels = radar.draw_range_labels(ax=ax, fontsize=15)
param_labels = radar.draw_param_labels(ax=ax, fontsize=17)



fig.text(
    0.85, 0.85,
    "Defending",
    size=25,
    ha="center", fontproperties=font_bold, color="black"
)

fig.text(
    0.875, 0.15,
    "Possession",
    size=25,
    ha="center", fontproperties=font_bold, color="black"
)

fig.text(
    0.175, 0.15,
    "Progression",
    size=25,
    ha="center", fontproperties=font_bold, color="black"
)


fig.text(
    0.175, 0.85,
    "Attacking",
    size=25,
    ha="center", fontproperties=font_bold, color="black"
)


fig.set_facecolor('#faf0e6')
plt.gca().set_facecolor('#faf0e6')

fig.set_dpi(600)

st.pyplot(fig)

