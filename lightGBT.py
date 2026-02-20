import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import cdist
import numpy as np
from datetime import date
import lightgbm as lgb


MAXIMUM_X = 120
MAXIMUM_Y = 53.3
MAXIMUM_DIS = 131.30
BASE_COLS = []

#The features s, a represent the spped and acceleration
#the s and a are calculated from r, where r = <x(t), y(t)>
#so the function is a vector valued function in terms of time t
#velocity is given by <x'(t), y'(t)>
#acceleration is given by <x"(t), y"(t)>
#the s and a given here are the magnitude of these vectors
#the absolute yardline number refers to the distance to score endzone of the offense, calculated from the line of scrimmage

BASE_COLS = ['game_id', 'play_id', 'player_to_predict', 'nfl_id', 'frame_id',
       'play_direction', 'absolute_yardline_number', 'player_name',
       'player_height', 'player_weight', 'player_birth_date',
       'player_position', 'player_side', 'player_role', 'x', 'y', 's', 'a',
       'dir', 'o', 'num_frames_output', 'ball_land_x', 'ball_land_y']

MANDATORY_COLS=['game_id', 'play_id', 'nfl_id']

MODEL_NUMERICAL_INPUTS = ['frame_id', 'absolute_yardline_number', 'player_height',
       'player_weight', 'x_last', 'y_last', 's', 'a', 'dir',
       'o', 'num_frames_output', 'ball_land_x', 'ball_land_y', 'x_prev',
       'y_prev', 'o_prev', 'dir_prev', 's_prev', 'a_prev',
       'nearest_offense_dis', 'nearest_offense_dis_x', 'nearest_offense_dis_y',
       'nearest_defense_dis', 'nearest_defense_dis_x', 'nearest_defense_dis_y',
       'receiver_x', 'receiver_y', 'height', 'velocity_x', 'velocity_y',
       'acc_x', 'acc_y', 'sin_o', 'cos_o', 'sin_dir', 'cos_dir', 'change_in_x',
       'change_in_y', 'change_in_s', 'change_in_a', 'change_in_o',
       'change_in_dir', 'dist_between_ball_land_and_player',
       'dist_x_between_ball_and_player', 'dist_y_between_ball_and_player',
       'angle_between_ball_and_dir', 'sin_angle_between_ball_and_dir',
       'cos_angle_between_ball_and_dir', 'angle_between_ball_and_o',
       'sin_angle_between_ball_and_o', 'cos_angle_between_ball_and_o',
       'distance_to_sideline', 'distance_to_receiver',
       'distance_x_to_receiver', 'distance_y_to_receiver',
       'angle_between_dir_and_receiver', 'sin_angle_between_dir_and_receiver',
       'cos_angle_between_dir_and_receiver', 'angle_between_o_and_receiver',
       'sin_angle_between_o_and_receiver', 'cos_angle_between_o_and_receiver',
       'time_left', 'required_speed', 'required_velocity_x',
       'required_velocity_y', 'required_acc_x', 'required_acc_y',
       'required_speed_diff', 'required_velocity_x_diff',
       'required_velocity_y_diff', 'required_acc_x_diff',
       'required_acc_y_diff', 'proj_x_acc', 'proj_y_acc', 'proj_x_velocity',
       'proj_y_velocity', 'proj_x_acc_diff', 'proj_y_acc_diff',
       'proj_x_velocity_diff', 'proj_y_velocity_diff', 'player_age']
                          
MODEL_CAT_INPUTS = ['player_role']

MODEL_OUTPUTS = ['dx', 'dy']

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

def reflect_input_player_positions(df):
    mask = df['play_direction'] == 'left'
    df.loc[mask, 'x'] = 120 - df.loc[mask, 'x']
    df.loc[mask, 'dir'] = (360 - df.loc[mask, 'dir']) % 360
    df.loc[mask, 'o'] = (360 - df.loc[mask, 'o']) % 360
    df.loc[mask, 'ball_land_x'] = 120 - df.loc[mask, 'ball_land_x']
    return df

def reflect_output_player_positions(df):
    mask = df['play_direction'] == 'left'
    df.loc[mask, 'x'] = 120 - df.loc[mask, 'x']
    return df


def add_nearest_dis_info(df):
    plays =  df.groupby(['game_id', 'play_id'], as_index=False)
    res = []
    for _, per_play in plays:
        per_play = per_play.copy()
        coords = per_play[['x_last', 'y_last']].to_numpy()
        if coords.shape[0] == 1:
            per_play['nearest_offense_dis'] = MAXIMUM_DIS
            per_play['nearest_defense_dis'] = MAXIMUM_DIS
            per_play['nearest_offense_dis_x'] = MAXIMUM_DIS
            per_play['nearest_offense_dis_y'] = MAXIMUM_DIS
            per_play['nearest_defense_dis_x'] = MAXIMUM_DIS
            per_play['nearest_defense_dis_y'] = MAXIMUM_DIS
        else:
            distance = cdist(coords, coords, metric='euclidean')
            np.fill_diagonal(distance, MAXIMUM_DIS)
                
            offense_indices = (per_play['player_side'] == 'Offense')
            defense_indices = (per_play['player_side'] == 'Defense')
                
            if np.any(offense_indices):
                per_play.loc[:,['nearest_offense_dis']] = np.min(distance[:,offense_indices], axis=-1)
                
                idx = np.argmin(distance[:,offense_indices], axis=-1)
                per_play.loc[:,['nearest_offense_dis_x']] = coords[offense_indices,][idx,0]
                per_play.loc[:,['nearest_offense_dis_y']] = coords[offense_indices,][idx,1]
            else:
                per_play.loc[:,['nearest_offense_dis']] = MAXIMUM_DIS
                per_play.loc[:,['nearest_offense_dis_x']] = MAXIMUM_X
                per_play.loc[:,['nearest_offense_dis_y']] = MAXIMUM_Y
    
            if np.any(defense_indices):
                per_play.loc[:,['nearest_defense_dis']] = np.min(distance[:,defense_indices],axis=-1)
                
                idx = np.argmin(distance[:,defense_indices], axis=-1)
                per_play.loc[:,['nearest_defense_dis_x']] = coords[defense_indices,][idx,0]
                per_play.loc[:,['nearest_defense_dis_y']] = coords[defense_indices][idx,1]
            else:
                per_play.loc[:,['nearest_defense_dis']] = MAXIMUM_DIS
                per_play.loc[:,['nearest_defense_dis_x']] = MAXIMUM_X
                per_play.loc[:,['nearest_defense_dis_y']] = MAXIMUM_Y
        
        res.append(per_play)
                
    return pd.concat(res, axis=0)


def add_reciever_info(df):
    receiver = df.loc[df['player_role'] == 'Targeted Receiver',['game_id', 'play_id', 'x_last', 'y_last']]

    receiver = receiver.rename(columns = {'x_last': 'receiver_x', 'y_last':'receiver_y'})

    #if a play has no targeted receiever we leave with nan
    return df.merge(receiver, on=['game_id', 'play_id'], how='left')


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

def clean_and_extract_features(df):
   
    def convert_to_radians(degrees):
        return degrees * np.pi / 180
        
    def convert_to_degrees(radians):
        return radians * 180 / np.pi

    def sin(theta):
        return np.sin(convert_to_radians(theta))

    def cos(theta):
        return np.cos(convert_to_radians(theta))

    def distance_between_two_points(x1, y1, x2, y2):
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def angle_between_two_vectors(x1, y1, x2, y2):
        denom = np.sqrt((x1**2)+(y1**2)) * np.sqrt((x2**2)+(y2**2)) + 1e-6
        dot = x1*x2 + y1*y2
        cos_angle = dot / denom
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_in_radians = np.arccos(cos_angle)
        return convert_to_degrees(angle_in_radians)

    def convert_to_inches(X):
        splits = X.str.split('-', expand=True)
        feet = splits.iloc[:,0].astype(np.float32)
        inches = splits.iloc[:,1].astype(np.float32)
        return feet * 12 + inches

    #returns the shortest angle when curr = 1degree prev = 359degree, change is 2degree
    def change_in_angle(curr, prev):
        return ((curr - prev +180) % 360) - 180
    
    df['x_last'] = df['x_last'].fillna(0.0)
    df['x_prev'] = df['x_prev'].fillna(0.0)
    df['y_last'] = df['y_last'].fillna(0.0)
    df['y_prev'] = df['y_prev'].fillna(0.0)
    df['s'] = df['s'].fillna(0.0)
    df['s_prev'] = df['s_prev'].fillna(0.0)
    df['a'] = df['a'].fillna(0.0)
    df['a_prev'] = df['a_prev'].fillna(0.0)
    df['dir'] = df['dir'].fillna(0.0)
    df['dir_prev'] = df['dir_prev'].fillna(0.0)
    df['o'] = df['o'].fillna(0.0)
    df['o_prev'] = df['o_prev'].fillna(0.0)
    df['receiver_x'] = df['receiver_x'].fillna(0.0)
    df['receiver_y'] = df['receiver_y'].fillna(0.0)
    df['player_height'] = df['player_height'].fillna('0-0')

    df['velocity_x'] = df['s'] * sin(df['dir'])
    df['velocity_y'] = df['s'] * cos(df['dir'])
    df['acc_x'] = df['a'] * sin(df['dir'])
    df['acc_y'] = df['a'] * cos(df['dir'])

    df['sin_o'] = sin(df['o'])
    df['cos_o'] = cos(df['o'])
    
    df['sin_dir'] = sin(df['dir'])
    df['cos_dir'] = cos(df['dir'])

    df['change_in_x'] = df['x_last'] - df['x_prev']
    df['change_in_y'] = df['y_last'] - df['y_prev']
    df['change_in_s'] = df['s'] - df['s_prev']
    df['change_in_a'] = df['a'] - df['a_prev']
    df['change_in_o'] = change_in_angle(df['o'], df['o_prev'])
    df['change_in_dir'] = change_in_angle(df['dir'], df['dir_prev'])
    
    df['dist_between_ball_land_and_player'] = distance_between_two_points(
        df['x_last'], df['y_last'], df['ball_land_x'], df['ball_land_y']
    )
    df['dist_x_between_ball_and_player'] = df['ball_land_x'] - df['x_last']
    df['dist_y_between_ball_and_player'] = df['ball_land_y'] - df['y_last']

    df['angle_between_ball_and_dir'] = angle_between_two_vectors(
        df['sin_dir'], df['cos_dir'], df['dist_x_between_ball_and_player'], df['dist_y_between_ball_and_player']
    )

    df['sin_angle_between_ball_and_dir'] = sin(df['angle_between_ball_and_dir'])
    df['cos_angle_between_ball_and_dir'] = cos(df['angle_between_ball_and_dir'])

    df['angle_between_ball_and_o'] = angle_between_two_vectors(
        df['sin_o'], df['cos_o'], df['dist_x_between_ball_and_player'], df['dist_y_between_ball_and_player']
    )
    
    df['sin_angle_between_ball_and_o'] = sin(df['angle_between_ball_and_o'])
    df['cos_angle_between_ball_and_o'] = cos(df['angle_between_ball_and_o'])
            
    df['distance_to_sideline'] = np.minimum(df['y_last'], MAXIMUM_Y - df['y_last'])

    df['player_height'] = convert_to_inches(df['player_height'])

    
    df['distance_to_receiver'] = distance_between_two_points(df['x_last'], df['y_last'], df['receiver_x'], df['receiver_y'])
    df['distance_x_to_receiver'] = df['x_last'] - df['receiver_x']
    df['distance_y_to_receiver'] = df['y_last'] - df['receiver_y']
    df['angle_between_dir_and_receiver'] = angle_between_two_vectors(
        df['sin_dir'], df['cos_dir'], df['distance_x_to_receiver'], df['distance_y_to_receiver']
    )

    df['sin_angle_between_dir_and_receiver'] = sin(df['angle_between_dir_and_receiver'])
    df['cos_angle_between_dir_and_receiver'] = cos(df['angle_between_dir_and_receiver'])

    df['angle_between_o_and_receiver'] = angle_between_two_vectors(df['sin_o'], df['cos_o'], df['distance_x_to_receiver'], df['distance_y_to_receiver'])

    df['sin_angle_between_o_and_receiver'] = sin(df['angle_between_o_and_receiver'])
    df['cos_angle_between_o_and_receiver'] = cos(df['angle_between_o_and_receiver'])
    

    df['time_left'] =  (df['num_frames_output'] - (df['frame_id'] - 1) )/10
        
    df['required_speed'] = df['dist_between_ball_land_and_player'] / df['time_left']
    df['required_velocity_x'] = df['dist_x_between_ball_and_player'] / df['time_left']
    df['required_velocity_y'] = df['dist_y_between_ball_and_player'] / df['time_left']
    df['required_acc_x'] = (df['required_velocity_x'] - df['velocity_x']) / df['time_left']
    df['required_acc_y'] = (df['required_velocity_y'] - df['velocity_y']) / df['time_left']

    df['required_speed_diff'] = df['required_speed'] - df['s']
    df['required_velocity_x_diff'] = df['required_velocity_x'] - df['velocity_x']
    df['required_velocity_y_diff'] = df['required_velocity_y'] - df['velocity_y']
    df['required_acc_x_diff'] = df['required_acc_x'] - df['acc_x']
    df['required_acc_y_diff'] = df['required_acc_y'] - df['acc_y']

    df['proj_x_acc'] = df['x_last'] + df['velocity_x']*df['time_left'] + 0.5*df['acc_x']*(df['time_left']**2)
    df['proj_y_acc'] = df['y_last'] + df['velocity_y']*df['time_left'] + 0.5*df['acc_y']*(df['time_left']**2)

    df['proj_x_velocity'] = df['x_last'] + df['velocity_x']*df['time_left']
    df['proj_y_velocity'] = df['y_last'] + df['velocity_y']*df['time_left']

    df['proj_x_acc_diff'] = df['ball_land_x'] - df['proj_x_acc']
    df['proj_y_acc_diff'] = df['ball_land_y'] - df['proj_y_acc']

    df['proj_x_velocity_diff'] = df['ball_land_x'] - df['proj_x_velocity']
    df['proj_y_velocity_diff'] = df['ball_land_y'] - df['proj_y_velocity']

    year = date.today().year
    s = pd.to_datetime(df['player_birth_date'])
    df['player_age'] = year - s.dt.year

    df['absolute_yardline_number'] = np.clip(df['absolute_yardline_number'], 0, 100.0)

    return df

def get_train_df(input_df, output_df):
    input_df = input_df.copy()
    input_df = reflect_input_player_positions(input_df)

    output_df = output_df.copy()
    
    df_last_frame = get_last_frame(input_df)
    df_last_frame = add_nearest_dis_info(df_last_frame)
    df_last_frame = add_reciever_info(df_last_frame)
    
    cols = ['game_id', 'play_id', 'nfl_id', 'absolute_yardline_number', 'player_height', 'player_weight', 'player_birth_date',
            'play_direction','player_position', 'player_role', 'x_last', 'y_last', 's', 'a', 'dir', 'o', 'num_frames_output', 'ball_land_x', 
            'ball_land_y', 'x_prev', 'y_prev', 'o_prev', 'dir_prev', 's_prev', 'a_prev', 'nearest_offense_dis', 'nearest_offense_dis_x', 
            'nearest_offense_dis_y', 'nearest_defense_dis', 'nearest_defense_dis_x', 'nearest_defense_dis_y', 'receiver_x', 'receiver_y']
    
    df_last_frame = df_last_frame[cols]
    
    df_merged = output_df.merge(df_last_frame, on=['game_id', 'play_id', 'nfl_id'], how='left')

    df_merged = reflect_output_player_positions(df_merged)
    
    X_df = clean_and_extract_features(df_merged)

    dx = X_df['x'] - X_df['x_last']
    dy = X_df['y'] - X_df['y_last']
    Y_df = pd.DataFrame({'dx' : dx, 'dy' :dy})

    return X_df, Y_df

def get_train_valid_split(X_df, Y_df, test_ratio=0.2):
    tr = X_df.shape[0]
    test_s = int(tr*(1-test_ratio))
    X_df_train = X_df[:test_s]
    Y_df_train = Y_df[:test_s]

    X_df_valid = X_df[test_s:]
    Y_df_valid = Y_df[test_s:]

    return X_df_train, Y_df_train, X_df_valid, Y_df_valid

def create_output_frames(df):
    num_frames = df['num_frames_output'].iloc[0]
    frame_id = pd.Series(np.arange(1, num_frames+1), name='frame_id')
    return df.merge(frame_id, how='cross')

def get_params(is_pred=True):
    
    params = {
                'task':'train', 
                'objective':'regression', 
                'bagging_freq' : 1,
                'bagging_fraction' : 0.75,
                'learning_rate' : 0.05,
                'device_type' : 'gpu',
                'num_threads':8,
                'n_estimators':2000,
                'seed' : 42,
                'max_depth':80,
                'max_leaves' : 100,
                'min_data_in_leaf' : 200,
                'feature_fraction' : 0.80,
                'lambda_l1' : 0.5,
                'lambda_l2' : 0.5,
                'early_stopping_rounds' : 100,
                'early_stopping_min_delta' : 0.001,
                'verbose':-1,
                'metric' :'rmse'
             }
    
    if is_pred:
        params['task'] = 'predict'

    return params

def get_test_df(input_df, output_df):
    df = input_df.copy()
    
    df = reflect_input_player_positions(df)
    df_last_frame = get_last_frame(df)
    df_last_frame = add_nearest_dis_info(df_last_frame)
    df_last_frame = add_reciever_info(df_last_frame)
    
    cols = ['game_id', 'play_id', 'nfl_id', 'absolute_yardline_number', 'player_height', 'player_weight', 'player_birth_date',
            'play_direction','player_role', 'player_position', 'x_last', 'y_last', 's', 'a', 'dir', 'o', 'num_frames_output', 'ball_land_x', 
            'ball_land_y', 'x_prev','y_prev', 'o_prev', 'dir_prev', 's_prev', 'a_prev', 'nearest_offense_dis', 'nearest_offense_dis_x', 
            'nearest_offense_dis_y', 'nearest_defense_dis', 'nearest_defense_dis_x', 'nearest_defense_dis_y', 'receiver_x', 'receiver_y']
    
    df_last_frame = df_last_frame[cols]

    df_merged = output_df.merge(df_last_frame, on=['game_id', 'play_id', 'nfl_id'], how='left')
    
    X_df = clean_and_extract_features(df_merged)

    X_df = X_df.drop(columns=['player_birth_date', 'game_id', 'play_id', 'nfl_id', 'x', 'y'])
    
    return X_df

def calculate_rmse(X_df, Y_df, dx_train, dy_train):
    pred_dx = dx_train.predict(X_df)
    pred_dy = dy_train.predict(X_df)

    n = Y_df.size
    residual = (pred_dx - Y_df['dx'])**2 + (pred_dy - Y_df['dy'])**2

    residual_avg = np.sum(residual) / n

    return np.sqrt(residual_avg)

def publish_results(X_df, Y_df, dx_train, dy_train):
    filt_cols = ['player_birth_date', 'x', 'y', 'game_id', 'play_id', 'nfl_id', 'play_direction', 'player_position']
    X_filt_df = X_df.drop(columns=filt_cols)
    X_filt_df = X_df.drop(columns=filt_cols)
    
    pred_dx = dx_train.predict(X_filt_df)
    pred_dy = dy_train.predict(X_filt_df)
    
    game_id = X_df['game_id'].values
    play_id = X_df['play_id'].values
    nfl_id =  X_df['nfl_id'].values
    frame_id = X_df['frame_id'].values
    player_position = X_df['player_position'].values
    player_role = X_df['player_role'].values
    x_last = X_df['x_last'].values
    y_last = X_df['y_last'].values
    actual_x = X_df['x'].values
    actual_y = X_df['y'].values
    play_direction = X_df['play_direction'].values

    x_last[play_direction == 'left'] = 120 - x_last[play_direction == 'left'] 
    actual_x[play_direction == 'left'] = 120 - actual_x[play_direction == 'left'] 

    pred_x = -pred_dx + x_last
    pred_y = pred_dy + y_last


    preds = np.column_stack([game_id, play_id, nfl_id, frame_id, player_position, 
                    player_role, play_direction, x_last, y_last, actual_x, actual_y, pred_x, pred_y])
    
        
    df = pd.DataFrame(preds, columns=['game_id', 'play_id', 'nfl_id','frame_id', 'player_position', 
                                    'player_role','play_direction', 
                                    'x_last', 'y_last', 'actual_x', 'actual_y', 'pred_x', 'pred_y'])
    
    df.to_csv('lightgbt_test_data_results.csv', index=False)
    print('results published')
    

def train(input_df, output_df):
    given_input_cols = set(input_df.columns)
    for c in MANDATORY_COLS:
        if c not in given_input_cols:
            raise Exception(f'{c} is missing in input_df')
        elif input_df[c].isna().any():
            raise Exception(f'{c} in input_df contains nan')
        
    
    given_output_cols = set(output_df.columns)
    for c in MANDATORY_COLS:
        if c not in given_output_cols:
            raise Exception(f'{c} is missing in output_df')
        elif output_df[c].isna().any():
            raise Exception(f'{c} in output_df contains nan')
    
    for c in BASE_COLS:
        if c not in input_df.columns:
            input_df[c] = np.nan
    
    print('getting train_df, valid_df')
    X_df, Y_df = get_train_df(input_df, output_df)
    
    for c in MODEL_CAT_INPUTS:
        X_df['player_role'] = X_df['player_role'].astype('category')
    
    print('splitting train test')
    X_train_df, Y_train_df, X_valid_df, Y_valid_df = get_train_valid_split(X_df, Y_df, test_ratio=0.05)

    filt_cols = ['player_birth_date', 'x', 'y', 'game_id', 'play_id', 'nfl_id', 'play_direction', 'player_position']
    X_train_filt_df = X_train_df.drop(columns=filt_cols)
    X_valid_filt_df = X_valid_df.drop(columns=filt_cols)
    
    print('training started')
    train_set_dx = lgb.Dataset(data=X_train_filt_df, label=Y_train_df['dx'])
    train_sub_set_dx = lgb.Dataset(data=X_train_filt_df[:50000], label=Y_train_df['dx'][:50000], reference=train_set_dx)
    valid_set_dx = lgb.Dataset(data=X_valid_filt_df, label=Y_valid_df['dx'], reference=train_set_dx)
    
    train_set_dy = lgb.Dataset(data=X_train_filt_df, label=Y_train_df['dy'])
    train_sub_set_dy = lgb.Dataset(data=X_train_filt_df[:50000], label=Y_train_df['dy'][:50000], reference=train_set_dy)
    valid_set_dy = lgb.Dataset(data=X_valid_filt_df, label=Y_valid_df['dy'], reference=train_set_dy)
    
    params = get_params(False)
    
    dx_train = lgb.train(params = params, 
                     train_set=train_set_dx, 
                     valid_sets=[train_sub_set_dx, valid_set_dx],
                     valid_names=['train', 'valid'],
                     callbacks=[
                         lgb.log_evaluation(period=10)
                     ]
                    )

    dy_train = lgb.train(params = params, 
                         train_set=train_set_dy, 
                         valid_sets=[train_sub_set_dy, valid_set_dy], 
                         valid_names=['train', 'valid'],
                         callbacks=[
                               lgb.log_evaluation(period=10) 
                         ]
                        )

    publish_results(X_valid_df, Y_valid_df, dx_train, dy_train)

    train_rmse = calculate_rmse(X_train_filt_df, Y_train_df, dx_train, dy_train)
    valid_rmse = calculate_rmse(X_valid_filt_df, Y_valid_df, dx_train, dy_train)

    print(f'rmse on the training set is :{train_rmse}')
    print(f'rmse on the validation set is :{valid_rmse}')
    return dx_train, dy_train

def get_features_by_importance(model):
    sort_index = np.argsort(-model.feature_importance(importance_type='gain'))
    return np.array(model.feature_name())[sort_index]


print('started')

input_df, output_df = get_input_output_df()

print('fetched input and output')

dx_train, dy_train = train(input_df, output_df)


dx_model_feature_importance = get_features_by_importance(dx_train)
dy_model_feature_importance = get_features_by_importance(dy_train)

print(f'top 15 features for dx_model: {dx_model_feature_importance[:15]}')
print(f'top 15 features for dy_model: {dy_model_feature_importance[:15]}')




