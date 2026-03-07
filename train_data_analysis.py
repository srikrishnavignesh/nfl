
from collections import Counter
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import os


TRAIN_INPUT_FILE_PATH = 'C:/Users/vigne/nfl/train_input'
TRAIN_OUTPUT_FILE_PATH = 'C:/Users/vigne/nfl/train_output'


def get_train_file_paths():
        
    def get_output_file(input_filename):
        return input_filename.replace('input', 'output')
    
    input_file_paths = []
    output_file_paths = []
    input_files_dir = TRAIN_INPUT_FILE_PATH
    for w in range(1, 19):
        input_filename = f'input_2023_w{w:02d}.csv'
        if  os.path.isfile(f'{input_files_dir}/{input_filename}'):
            output_filename = get_output_file(input_filename)
            input_file_path = os.path.join(TRAIN_INPUT_FILE_PATH, input_filename)
            output_file_path = os.path.join(TRAIN_OUTPUT_FILE_PATH, output_filename)
            input_file_paths.append(input_file_path)
            output_file_paths.append(output_file_path)
        else:
            raise Exception(f'input file for week {w} does not exist')
            
    return (input_file_paths, output_file_paths)

def load_file(file_path):
    return pd.read_csv(file_path)
    
def get_input_output_df():
    input_file_paths, output_file_paths = get_train_file_paths()
   
    with ThreadPoolExecutor(max_workers = 8) as executor:
        input_dfs = executor.map(load_file, input_file_paths)
        input_df = pd.concat(input_dfs, axis=0)

    with ThreadPoolExecutor(max_workers = 8) as executor:
        output_dfs = executor.map(load_file, output_file_paths)
        output_df = pd.concat(output_dfs, axis=0)

    return input_df.reset_index(drop=True), output_df.reset_index(drop=True)

def plot_distribution_of_features(input_df, output_df):
    predict_players_position = Counter()
    predict_players_role = Counter()
    predict_players_side = Counter()
    
    num_frames_to_predict = Counter()
    no_players_prediction_in_a_play = Counter()
    

    plays = input_df.groupby(['game_id', 'play_id'], as_index = False)
    
    per_play_change_in_dists = []
    per_frame_change_in_dists = []
    per_frame_change_in_x_dists = []
    per_frame_change_in_y_dists = []
    for _, play in plays:

        predict_players = play[play['player_to_predict']]
        
        no_players_prediction_in_a_play[predict_players['nfl_id'].nunique()]+=1

        
        num_frames_output = play['num_frames_output'].iloc[0].item()
        num_frames_to_predict[num_frames_output]+=1
        
        predict_players_last_frame = predict_players.groupby(['nfl_id'], as_index=False).last()
        for index, p_l in predict_players_last_frame.iterrows():
            game_id = p_l['game_id']
            play_id = p_l['play_id']
            p_nfl_id = p_l['nfl_id']
            p_output = output_df[(output_df['game_id'] == game_id) & (output_df['play_id'] == play_id) & (output_df['nfl_id'] == p_nfl_id)]
            
            
            s = np.array([p_l['x'], p_l['y']])
            
            total_dis = 0
            for _,p_o in p_output.iterrows():
                e = np.array([p_o['x'].item(), p_o['y'].item()])
                current_dis = np.linalg.norm(e - s)
                per_frame_change_in_dists.append(current_dis)
                per_frame_change_in_x_dists.append(np.abs(e[0] - s[0]))
                per_frame_change_in_y_dists.append(np.abs(e[1] - s[1]))
            
                total_dis+= current_dis
                s = e
    
            per_play_change_in_dists.append(total_dis)
    
    
            position = p_l['player_position']
            role = p_l['player_role']
            side = p_l['player_side']
            
            predict_players_position[position]+=1
            predict_players_role[role]+=1
            predict_players_side[side]+=1
            
    def plot_bargraph(dict_items, name):
        plt.figure(figsize=(10, 6))
        df = pd.DataFrame(list(dict_items), columns = [name, 'count'])
        df = df.sort_values(by='count', ascending=False)
        sns.barplot(data=df, x=name, y='count')
        plt.show()

    
    plot_bargraph(predict_players_position.items(), 'predict_player position')
    plot_bargraph(predict_players_role.items(), 'predict_player role')
    plot_bargraph(predict_players_side.items(), 'predict_player side')

    plot_bargraph(no_players_prediction_in_a_play.items(), 'num of player to predict in a play')
    plot_bargraph(num_frames_to_predict.items(), 'num of frames to predict in a play')


    def plot_density_plot(data, title, x):
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data, fill=True, color="dodgerblue")
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel('Density')
        plt.show()
    
   
    plot_density_plot(per_play_change_in_dists, 'total distance moved by a player in a play', 'distance')
    plot_density_plot(per_frame_change_in_dists, 'distance moved by a player per frame', 'distance')
    plot_density_plot(per_frame_change_in_x_dists, 'distance moved by a player along x per frame', 'distance')
    plot_density_plot(per_frame_change_in_y_dists, 'distance moved by a player along y per frame', 'distance')


def get_last_frame(df):
    
    df_sorted = df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id']).reset_index(drop=True)
    
    group_by_cols = ['game_id', 'play_id', 'nfl_id']

    feature_cols = ['x', 'y', 'o', 'dir', 's', 'a']
    
    df_sorted[[f'{c}_prev' for c in feature_cols]] = df_sorted.groupby(group_by_cols)[feature_cols].shift(1)

    #last() takes non none values from the last possible col
    #so even if last frame misses a feature , value is taken from the previous available one
    df_last_frame = df_sorted.groupby(group_by_cols, as_index=False).last()

    df_last_frame = df_last_frame.rename(columns={'x':'x_last', 'y':'y_last'})
    
    return df_last_frame


def predict_physics_baseline(input_df, output_df):

    def convert_to_radians(degrees):
        return degrees * np.pi / 180

    def sin(theta):
        return np.sin(convert_to_radians(theta))
    
    def cos(theta):
        return np.cos(convert_to_radians(theta))
    

    input_df = input_df.copy()
    
    output_df = output_df.copy()
    
    df_last_frame = get_last_frame(input_df)

    df_last_frame = df_last_frame[['game_id', 'play_id', 'nfl_id', 'x_last', 'y_last', 'o', 'dir', 's', 'a', 'num_frames_output']]
    
    df = output_df.merge(df_last_frame, on=['game_id', 'play_id', 'nfl_id'], how='left')

    sum_ = 0
    for _, group_df in df.groupby(['game_id', 'play_id', 'nfl_id'], as_index=False):

        group_df = group_df.sort_values('frame_id').reset_index(drop=True)

        prev = (group_df.iloc[0]['x'], group_df.iloc[0]['y'])
        for row in group_df.itertuples():
            dt = 0.1

            velocity_x = row.s * sin(row.dir)
            velocity_y_ = row.s * cos(row.dir)
            acc_x_ = row.a * sin(row.dir)
            acc_y_ = row.a * cos(row.dir)

            proj_x = prev[0] + velocity_x*dt + 0.5*acc_x_*(dt**2)
            proj_y = prev[1] + velocity_y_*dt + 0.5*acc_y_*(dt**2)
            
            sum_+= (row.x - proj_x)**2 + (row.y - proj_y)**2
            prev = (proj_x, proj_y)
       
    num_ele = df.shape[0]*2
    rmse = np.sqrt(sum_ / num_ele)
    print(f'RMSE of the simple physics based model is {rmse}')


input_df, output_df = get_input_output_df()
POSITION_MAPPING = [
    "FS --> Free Safety",
    "SS --> Strong Safety",
    "CB --> Cornerback",
    "MLB --> Middle Linebacker",
    "WR --> Wide Receiver",
    "TE --> Tight End",
    "QB --> Quarterback",
    "OLB --> Outside Linebacker",
    "ILB --> Inside Linebacker",
    "RB --> Running Back",
    "DE --> Defensive End",
    "FB --> Fullback",
    "NT --> Nose Tackle",
    "DT --> Defensive Tackle",
    "S --> Safety",
    "T --> Tackle",
    "LB --> Linebacker",
    "P --> Punter",
    "K --> Kicker"
]

PLAYER_ROLES = ['Defensive Coverage' 'Other Route Runner' 'Passer' 'Targeted Receiver']

PLAYER_SIDES = ['Defense', 'Offense']

print(f'player positions are {POSITION_MAPPING}')
print(f'player roles are {PLAYER_ROLES}')
print(f'player roles are {PLAYER_SIDES}')

# plot_distribution_of_features(input_df[:1_000_00], output_df)

predict_physics_baseline(input_df, output_df)