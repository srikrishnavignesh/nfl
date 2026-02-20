import numpy as np
import pandas as pd
import torch


MODEL_INPUT_FEATURES = ['frame_id', 'absolute_yardline_number', 
       'player_height', 'player_weight', 'x', 'y', 's', 'a', 'dir', 'o',
       'num_frames_output', 'ball_land_x', 'ball_land_y', 'sin_dir', 'cos_dir',
       'sin_o', 'cos_o', 'velocity_x', 'velocity_y', 'acc_x', 'acc_y',
       'distance_to_sideline', 'distance_to_ball_land',
       'distance_to_ball_land_x', 'distance_to_ball_land_y',
       'angle_between_ball_land_and_player',
       'angle_between_ball_land_and_player_orient',
       'sin_angle_between_ball_land_and_player',
       'cos_angle_between_ball_land_and_player',
       'sin_angle_between_ball_land_and_player_orient',
       'cos_angle_between_ball_land_and_player_orient', 'time_left',
       'required_speed', 'required_acceleration', 'change_in_x', 'change_in_y',
       'change_in_speed', 'change_in_acceleration', 'change_in_o',
       'sin_change_in_o', 'cos_change_in_o', 'change_in_dir',
       'sin_change_in_dir', 'cos_change_in_dir',
       'distance_between_the_reciever', 'distance_between_the_passer',
       'distance_to_defense', 'distance_to_defense_x', 'distance_to_defense_y',
       'distance_to_offense', 'distance_to_offense_x', 'distance_to_offense_y',
       'nearest_teammate_dis', 'nearest_teammate_dis_x',
       'nearest_teammate_dis_y', 'player_role_Defensive Coverage',
       'player_role_Other Route Runner', 'player_role_Passer',
       'player_role_Targeted Receiver', 'player_role_unknown',
       'required_velocity_x', 'required_velocity_y', 'required_acc_x',
       'required_acc_y', 'required_speed_diff', 'required_velocity_x_diff',
       'required_velocity_y_diff', 'required_acc_x_diff',
       'required_acc_y_diff', 'proj_x_acc', 'proj_y_acc', 'proj_x_velocity',
       'proj_y_velocity', 'proj_x_acc_diff', 'proj_y_acc_diff',
       'proj_x_velocity_diff', 'proj_y_velocity_diff',
       'angle_between_orientation_and_player', 'player_age']

MODEL_OUTPUTS = ['x', 'y']

FILE_PATH = 'C:/Users/vigne/nfl'
TRAIN_INPUT_FILE_PATH=f'{FILE_PATH}/train_input'
TRAIN_OUTPUT_FILE_PATH=f'{FILE_PATH}/train_output'

DX_CHECKPOINT_DIR = f'{FILE_PATH}/checkpoints_dx'
DY_CHECKPOINT_DIR = f'{FILE_PATH}/checkpoints_dy'

BEST_DX_MODEL_CHECKPOINT = f'{FILE_PATH}/cross_attn_best_weight_dx.pt'
BEST_DY_MODEL_CHECKPOINT = f'{FILE_PATH}/cross_attn_best_weight_dy.pt'

JOBLIB_FILE_PATH = f'{FILE_PATH}/gru_scaler.joblib'


from enum import Enum
class ModelType(Enum):
    DX_MODEL = 1
    DY_MODEL = 2

from datetime import date
from scipy.spatial.distance import cdist

class NFLFeatureTransformer:

    def _convert_to_radians(self, degrees):
        return degrees * np.pi / 180
        
    def _convert_to_degrees(self, radians):
        return radians * 180 / np.pi

    def _sin(self, X):
        return np.sin(self._convert_to_radians(X))

    def _cos(self, X):
        return np.cos(self._convert_to_radians(X))

    def _x_component(self, magnitude, direction):
        return magnitude * self._sin(direction)

    def _y_component(self, magnitude, direction):
        return magnitude * self._cos(direction)

    def _distance_between_two_points(self, x1, y1, x2, y2):
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

   
    def _angle_between_two_points(self, x1, y1, x2, y2):
        num = x2 - x1
        denom = (y2-y1)+1e-6 
        angle_in_radians = np.arctan2(num, denom)
        angle = self._convert_to_degrees(angle_in_radians)
        return (angle + 360) % 360
    
    def _angle_between_two_vectors(self, x1, y1, x2, y2):
        denom = np.sqrt((x1**2)+(y1**2)) * np.sqrt((x2**2)+(y2**2)) + 1e-6
        dot = x1*x2 + y1*y2
        cos_angle = dot / denom
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_in_radians = np.arccos(cos_angle)
        return self._convert_to_degrees(angle_in_radians)

    def _convert_to_inches(self, X):
        splits = X.str.split('-', expand=True)
        feet = splits.iloc[:,0].astype(np.float32)
        inches = splits.iloc[:,1].astype(np.float32)
        return feet * 12 + inches
    
    MAXIMUM_X = 120
    MAXIMUM_Y = 53.3

    #measured along diagonal
    MAXIMUM_DIS = 131.30

    #measures short change in angle like (1 degree - 359 degree gives -358 degree/360 gives a larger value)
    def _change_in_angle(self, prev, curr):
        return ((curr - prev +180) % 360) - 180
    
    def reflect_input_coordinates(self, X, play_direction):
        if play_direction == 'left':
            X['x'] = self.MAXIMUM_X - X['x']
            X['dir'] = (360 - X['dir']) % 360
            X['o'] = (360 - X['o']) % 360
            X['ball_land_x'] = self.MAXIMUM_X - X['ball_land_x']
        return X
    
    def reflect_output_coordinates(self, Y, play_direction):
        if play_direction == 'left':
            Y['x'] = self.MAXIMUM_X - Y['x']
        return Y
    
    
    def transform_X(self, X, prev_frame, total_frames):

        X['x'] = X['x'].fillna(0.0)
        X['y'] = X['y'].fillna(0.0)
        X['s'] = X['s'].fillna(0.0)
        X['a'] = X['a'].fillna(0.0)
        X['dir'] = X['dir'].fillna(0.0)
        X['o'] = X['o'].fillna(0.0)
        X['player_height'] = X['player_height'].fillna('0-0')

        prev_x_ = prev_frame['x'].fillna(0.0).values
        prev_y_ = prev_frame['y'].fillna(0.0).values
        prev_s_ = prev_frame['s'].fillna(0.0).values
        prev_a_ = prev_frame['a'].fillna(0.0).values
        prev_dir_ = prev_frame['dir'].fillna(0.0).values
        prev_o_ = prev_frame['o'].fillna(0.0).values

        player_role_categories = ['Defensive Coverage', 'Other Route Runner', 'Passer', 'Targeted Receiver', 'unknown']
        X['player_role'] = X['player_role'].fillna('unknown')
        X['player_role'] = pd.Categorical(X['player_role'], categories=player_role_categories)

        offense_indices = X['player_side'] == 'Offense'
        defense_indices = X['player_side'] == 'Defense'
        targeted_reciever = X[X['player_role'] == 'Targeted Receiver']
        passer = X[X['player_role'] == 'Passer']

        if np.any(defense_indices):
            defense_x_ = X[defense_indices]['x'].values
            defense_y_ = X[defense_indices]['y'].values
            defense_pos = np.column_stack((defense_x_, defense_y_))

        if np.any(offense_indices):
            offense_x_ = X[offense_indices]['x'].values
            offense_y_ = X[offense_indices]['y'].values
            offense_pos = np.column_stack((offense_x_, offense_y_))

            
        X['sin_dir'] = self._sin(X['dir'])
        X['cos_dir'] = self._cos(X['dir'])
        X['sin_o'] = self._sin(X['o'])
        X['cos_o'] = self._cos(X['o'])
        X['velocity_x'] = self._x_component(X['s'], X['dir']) 
        X['velocity_y'] = self._y_component(X['s'], X['dir']) 

        X['acc_x'] = self._x_component(X['a'], X['dir']) 
        X['acc_y'] = self._y_component(X['a'], X['dir']) 

        X['distance_to_sideline'] = np.minimum(X['y'], self.MAXIMUM_Y - X['y'])
        

        X['distance_to_ball_land'] = self._distance_between_two_points(X['x'], X['y'], X['ball_land_x'],X['ball_land_y']) 
        X['distance_to_ball_land_x'] = X['x'] - X['ball_land_x']
        X['distance_to_ball_land_y'] = X['y'] - X['ball_land_y']

        X['angle_between_ball_land_and_player'] = (
            self._angle_between_two_vectors(X['sin_dir'], X['cos_dir'], X['distance_to_ball_land_x'], X['distance_to_ball_land_y'] ) 
        )

        X['angle_between_ball_land_and_player_orient'] = (
            self._angle_between_two_vectors(X['sin_o'], X['cos_o'], X['distance_to_ball_land_x'], X['distance_to_ball_land_y'] )
        )

        X['sin_angle_between_ball_land_and_player'] = self._sin(X['angle_between_ball_land_and_player'])
        X['cos_angle_between_ball_land_and_player'] = self._cos(X['angle_between_ball_land_and_player'])

        X['sin_angle_between_ball_land_and_player_orient'] = self._sin(X['angle_between_ball_land_and_player_orient'])
        X['cos_angle_between_ball_land_and_player_orient'] = self._cos(X['angle_between_ball_land_and_player_orient'])

        
        X['time_left'] = (total_frames - (X['frame_id'] - 1)) / 10
        X['required_speed'] = (X['distance_to_ball_land'] / X['time_left']) 
        X['required_acceleration'] = ((X['required_speed'] - X['s']) / X['time_left'])

        
        X['change_in_x'] = (X['x'] - prev_x_) 
        X['change_in_y'] = (X['y'] - prev_y_) 
        X['change_in_speed'] = (X['s'] - prev_s_) 
        X['change_in_acceleration'] = (X['a'] - prev_a_)

        X['change_in_o'] = self._change_in_angle(X['o'], prev_o_)
        X['sin_change_in_o'] = self._sin(X['change_in_o'])
        X['cos_change_in_o'] = self._cos(X['change_in_o'])
        
        X['change_in_dir'] = self._change_in_angle(X['dir'] , prev_dir_)
        X['sin_change_in_dir'] = self._sin(X['change_in_dir'])
        X['cos_change_in_dir'] = self._cos(X['change_in_dir'])

        
        if targeted_reciever.empty:
            X['distance_between_the_reciever'] = self.MAXIMUM_DIS
        else:
            X['distance_between_the_reciever'] = (
                self._distance_between_two_points(X['x'], X['y'], targeted_reciever['x'].values, targeted_reciever['y'].values) 
            )

        if passer.empty:
            X['distance_between_the_passer'] = self.MAXIMUM_DIS
        else:
            X['distance_between_the_passer'] = self._distance_between_two_points(X['x'], X['y'], passer['x'].values, passer['y'].values) 

        
        if np.any(offense_indices) and np.any(defense_indices):
            cross_distance_matrix = cdist(offense_pos, defense_pos, metric='euclidean')
            
            offense_distance_matrix = cdist(offense_pos, offense_pos, metric='euclidean')
            np.fill_diagonal(offense_distance_matrix, self.MAXIMUM_DIS)
            
            defense_distance_matrix = cdist(defense_pos, defense_pos, metric='euclidean')
            np.fill_diagonal(defense_distance_matrix, self.MAXIMUM_DIS)
            
            X.loc[offense_indices, ['distance_to_defense']] = np.min(cross_distance_matrix, axis=1)
            X.loc[offense_indices, ['distance_to_defense_x']] = defense_pos[np.argmin(cross_distance_matrix, axis=1), 0]
            X.loc[offense_indices, ['distance_to_defense_y']] = defense_pos[np.argmin(cross_distance_matrix, axis=1), 1]

            X.loc[offense_indices, ['distance_to_offense']] = 0
            X.loc[offense_indices, ['distance_to_offense_x']] = 0
            X.loc[offense_indices, ['distance_to_offense_y']] = 0
            
            X.loc[offense_indices, ['nearest_teammate_dis']] = np.min(offense_distance_matrix, axis=1)
            X.loc[offense_indices, ['nearest_teammate_dis_x']] = offense_pos[np.argmin(offense_distance_matrix, axis=1), 0]
            X.loc[offense_indices, ['nearest_teammate_dis_y']] = offense_pos[np.argmin(offense_distance_matrix, axis=1), 1]
            

        
            X.loc[defense_indices, ['distance_to_offense']] = np.min(cross_distance_matrix, axis=0)
            X.loc[defense_indices, ['distance_to_offense_x']] = offense_pos[np.argmin(cross_distance_matrix, axis=0), 0]
            X.loc[defense_indices, ['distance_to_offense_y']] = offense_pos[np.argmin(cross_distance_matrix, axis=0), 1]

            X.loc[defense_indices, ['distance_to_defense']] = 0
            X.loc[defense_indices, ['distance_to_defense_x']] = 0 
            X.loc[defense_indices, ['distance_to_defense_y']] = 0 

            X.loc[defense_indices, ['nearest_teammate_dis']] = np.min(defense_distance_matrix, axis=1)
            X.loc[defense_indices, ['nearest_teammate_dis_x']] = defense_pos[np.argmin(defense_distance_matrix, axis=1), 0]
            X.loc[defense_indices, ['nearest_teammate_dis_y']] = defense_pos[np.argmin(defense_distance_matrix, axis=1), 1]
        else:
            if np.any(offense_indices):
                offense_distance_matrix = cdist(offense_pos, offense_pos, metric='euclidean')
                np.fill_diagonal(offense_distance_matrix, self.MAXIMUM_DIS)
                X.loc[offense_indices, ['distance_to_defense']] = self.MAXIMUM_DIS
                X.loc[offense_indices, ['distance_to_defense_x']] = self.MAXIMUM_DIS
                X.loc[offense_indices, ['distance_to_defense_y']] = self.MAXIMUM_DIS
                
                X.loc[offense_indices, ['distance_to_offense']] = 0
                X.loc[offense_indices, ['distance_to_offense_x']] = 0
                X.loc[offense_indices, ['distance_to_offense_y']] = 0
               
                X.loc[offense_indices, ['nearest_teammate_dis']] = np.min(offense_distance_matrix, axis=1)
                X.loc[offense_indices, ['nearest_teammate_dis_x']] = offense_pos[np.argmin(offense_distance_matrix, axis=1), 0]
                X.loc[offense_indices, ['nearest_teammate_dis_y']] = offense_pos[np.argmin(offense_distance_matrix, axis=1), 1]
            
            if np.any(defense_indices):
                defense_distance_matrix = cdist(defense_pos, defense_pos, metric='euclidean')
                np.fill_diagonal(defense_distance_matrix, self.MAXIMUM_DIS)
                X.loc[defense_indices, ['distance_to_offense']] = self.MAXIMUM_DIS
                X.loc[defense_indices, ['distance_to_offense_x']] = self.MAXIMUM_DIS
                X.loc[defense_indices, ['distance_to_offense_y']] = self.MAXIMUM_DIS

                X.loc[defense_indices, ['distance_to_defense']] = 0
                X.loc[defense_indices, ['distance_to_defense_x']] = 0
                X.loc[defense_indices, ['distance_to_defense_y']] = 0
                
                X.loc[defense_indices, ['nearest_teammate_dis']] =  np.min(defense_distance_matrix, axis=1)
                X.loc[defense_indices, ['nearest_teammate_dis_x']] = defense_pos[np.argmin(defense_distance_matrix, axis=1), 0]
                X.loc[defense_indices, ['nearest_teammate_dis_y']] = defense_pos[np.argmin(defense_distance_matrix, axis=1), 1]
        
        
        X = pd.get_dummies(X, columns=['player_role'], dtype=np.float32)

        X['required_velocity_x'] = X['distance_to_ball_land_x'] / X['time_left']
        X['required_velocity_y'] = X['distance_to_ball_land_y'] / X['time_left']
        
        X['required_acc_x'] = (X['required_velocity_x'] - X['velocity_x']) / X['time_left']
        X['required_acc_y'] = (X['required_velocity_y'] - X['velocity_y']) / X['time_left']

        X['required_speed_diff'] = X['required_speed'] - X['s']
        X['required_velocity_x_diff'] = X['required_velocity_x'] - X['velocity_x']
        X['required_velocity_y_diff'] = X['required_velocity_y'] - X['velocity_y']
        
        X['required_acc_x_diff'] = X['required_acc_x'] - X['acc_x']
        X['required_acc_y_diff'] = X['required_acc_y'] - X['acc_y']

        X['proj_x_acc'] = X['x'] + X['velocity_x']*X['time_left'] + 0.5*X['acc_x']*(X['time_left']**2)
        X['proj_y_acc'] = X['y'] + X['velocity_y']*X['time_left'] + 0.5*X['acc_y']*(X['time_left']**2)

        X['proj_x_velocity'] = X['x'] + X['velocity_x']*X['time_left']
        X['proj_y_velocity'] = X['y'] + X['velocity_y']*X['time_left']

        X['proj_x_acc_diff'] = X['ball_land_x'] - X['proj_x_acc']
        X['proj_y_acc_diff'] = X['ball_land_y'] - X['proj_y_acc']

        X['proj_x_velocity_diff'] = X['ball_land_x'] - X['proj_x_velocity']
        X['proj_y_velocity_diff'] = X['ball_land_y'] - X['proj_y_velocity']


        X['angle_between_orientation_and_player'] = self._change_in_angle(X['o'], X['dir'])
        year = date.today().year
        s = pd.to_datetime(X['player_birth_date'])
        X['player_age'] = year - s.dt.year
        X['player_height'] = self._convert_to_inches(X['player_height']) 
        X['player_weight'] = X['player_weight']

        X['absolute_yardline_number'] = np.clip(X['absolute_yardline_number'], 0, 100.0)

        return X

    
from torch.utils.data import Dataset, get_worker_info
import os
import random

class NFLDataset(Dataset):
    def __init__(self, input_groups, output_groups, nfl_feature_transformer):
        self.nfl_feature_transformer = nfl_feature_transformer
        self.input_groups = input_groups
        self.output_groups = output_groups
        
    def _get_output_array(self, dx, dy):
        numpy_array = np.column_stack((dx, dy)).astype(np.float32)
        return np.expand_dims(numpy_array, axis=1)

    def _get_input_array(self, df):
        model_input_features = MODEL_INPUT_FEATURES
        numpy_array = df[model_input_features].to_numpy().astype(np.float32)
        return np.expand_dims(numpy_array, axis=1)

        
    def __len__(self):
        return len(self.input_groups)

    def __getitem__(self, indx):
        
        input_df = self.input_groups[indx].copy()
        output_df = self.output_groups[indx].copy()

        play_direction = input_df.iloc[0]['play_direction']

        input_df = self.nfl_feature_transformer.reflect_input_coordinates(input_df, play_direction)
        output_df = self.nfl_feature_transformer.reflect_output_coordinates(output_df, play_direction)

        total_frames = input_df['frame_id'].max() + input_df['num_frames_output'].iloc[0]
        
        prev_frame = None
        x = None
        for _, grouped_frame in input_df.sort_values(by=['frame_id', 'nfl_id'], ascending=[True, True]).groupby(by=['frame_id'], 
                                                                                                          as_index=False, sort=False):
            prev_frame = grouped_frame if prev_frame is None else prev_frame
            transformed_input_df = self.nfl_feature_transformer.transform_X(grouped_frame, prev_frame, total_frames)
            input_array = self._get_input_array(transformed_input_df)
            x = input_array if x is None else np.concat((x, input_array), axis=1)
            prev_frame = grouped_frame
        
        
        num_output = prev_frame['num_frames_output'].iloc[0] # type: ignore
        players_to_predict = prev_frame['player_to_predict'].values.tolist() # type: ignore

        x_last = prev_frame[prev_frame['player_to_predict']]['x'].values # type: ignore
        y_last = prev_frame[prev_frame['player_to_predict']]['y'].values # type: ignore

        y = None
        for _, grouped_frame in output_df.sort_values(by=['frame_id', 'nfl_id'], ascending=[True, True]).groupby(by=['frame_id'], 
                                                                                                            as_index=False, sort=False):
            
            dx = grouped_frame['x'] - x_last
            dy = grouped_frame['y'] - y_last
            
            output_tensor = self._get_output_array(dx, dy)
            y = output_tensor if y is None else np.concat((y, output_tensor), axis=1)
            
        return x, y, players_to_predict, num_output
    
    
import os
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

import random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

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


import random
def split_train_test(input_df, output_df, nfl_feature_transformer, test_ratio = 0.2):
    
    input_groups = [group for _, group in input_df.groupby(by=['game_id', 'play_id'], sort=True)]
    output_groups = [group for _, group in output_df.groupby(by=['game_id', 'play_id'], sort=True)]

    n_groups = len(input_groups)

    test_start = int(n_groups - n_groups*(test_ratio))

    train_input_groups, train_output_groups = input_groups[:test_start], output_groups[:test_start]
    train_data = NFLDataset(train_input_groups, train_output_groups, nfl_feature_transformer)
    

    test_input_groups, test_output_groups =  input_groups[test_start:], output_groups[test_start:]
    test_data = NFLDataset(test_input_groups, test_output_groups, nfl_feature_transformer)
    
    return train_data, test_data



# %%
from torch.nn import TransformerEncoder, TransformerEncoderLayer, GRU, Linear, Module, MSELoss, MultiheadAttention, Sequential, Linear, ReLU, Dropout
from positional_encodings.torch_encodings import PositionalEncoding1D

class SpatialTemporalLayer(Module):
    def __init__(self, d_model, nhead, spatial_encoder_layers, dropout, in_features):

        super().__init__()

        features_to_predict = 2
        dim_feedforward = 2*d_model
       
        transformer_encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
                                                            dropout=dropout, batch_first=True)

        self.se_input_projection = Linear(in_features = in_features, out_features = d_model)
        
        self.spatial_encoder = TransformerEncoder(encoder_layer=transformer_encoder_layer,enable_nested_tensor=False, num_layers=spatial_encoder_layers)
        
        self.temporal_encoder = GRU(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)
        self.temporal_decoder = GRU(input_size=features_to_predict, hidden_size=d_model, num_layers=1, batch_first=True)
        self.td_output_projection = Linear(in_features = d_model, out_features = features_to_predict)

        self.cross_attn = MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.gate = Linear(2 * d_model, d_model)

        self.mlp = Sequential(
                            Linear(d_model, 2*d_model),
                            ReLU(),
                            Linear(2*d_model, 2*d_model),
                            ReLU(),
                            Linear(2*d_model, 1)
                )

        self.mse_loss = MSELoss(reduction='mean')
        self.sum_se = MSELoss(reduction='sum')

        self.pos_enc = PositionalEncoding1D(d_model)
        

    #x must be [P, T, din] T-> time in a play_id , P --> number of players of that play_id, din --> input_feature
    #y must be in [P, T, 2]
    def forward(self, x, y, predict_bool, num_output):

        #converts to d_model
        x_projected = self.se_input_projection(x)

        x_permuted = x_projected.permute(1, 0, 2)
        
        se_output = self.spatial_encoder(x_permuted)

        predict_players_x = x_projected[predict_bool, :, :]
        
        no_players = predict_players_x.shape[0]
        d_model = predict_players_x.shape[2]
        time = predict_players_x.shape[1]
        
        hidden_state = torch.zeros(1, no_players, d_model, device=predict_players_x.device)
        for t in range(time):
            query = predict_players_x[:,t, :]
            key = se_output[t, :,:]
            value = se_output[t, :,:]
            
            cross_attn_output = self.cross_attn(query, key, value, need_weights=False)[0].unsqueeze(0)

            combined = torch.cat([hidden_state, cross_attn_output], dim=-1)   
            g = torch.sigmoid(self.gate(combined))

            hidden_state = g * hidden_state + (1 - g) * cross_attn_output

            gru_output, hidden_state = self.temporal_encoder(predict_players_x[:,t:t+1, :], hidden_state)
           
        time_expanded_state = gru_output.repeat(1, num_output, 1)
        encodings = self.pos_enc(time_expanded_state)
        encodings_added = time_expanded_state + encodings

        preds = self.mlp(encodings_added)
       
        preds_reshaped = preds.reshape(-1, 1)
        preds_detached=preds_reshaped.detach()

        if y is None:
            return preds_detached
        
        actual_reshaped = y.reshape(-1, 1)
        
        return self.mse_loss(preds_reshaped, actual_reshaped), self.sum_se(preds_reshaped, actual_reshaped), preds_detached

def checkpoint_state(epoch, model, optimizer, checkpoint_dir):
    directory = checkpoint_dir
    checkpoint = {'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}

    new_file_name = f'checkpoint-{epoch}.pt'
    old_file_name = f'checkpoint-{epoch - 1}.pt'

    # flush previous checkpoints
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path) and item.startswith('checkpoint-') and item != old_file_name:
            os.remove(item_path)

    file_path = os.path.join(directory, new_file_name)

    torch.save(checkpoint, file_path)

def setup_state_from_checkpoint(model, optimizer, checkpoint_dir, best_model_checkpoint_loc, device='cuda'):
    latest_epoch = -1
    directory = checkpoint_dir
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path) and item.startswith('checkpoint-'):
            epoch = int(item.removeprefix('checkpoint-').removesuffix('.pt'))
            latest_epoch = max(epoch, latest_epoch)

    if latest_epoch == -1:
        print('no existing checkpoint found')
        return 0, math.inf

    print(f'resuming states with {latest_epoch} checkpoint')
    checkpoint_pt = torch.load(os.path.join(directory, f'checkpoint-{latest_epoch}.pt'), map_location=device)
    model.load_state_dict(checkpoint_pt['model_state'])
    if 'optimizer_state' in checkpoint_pt:
        optimizer.load_state_dict(checkpoint_pt['optimizer_state'], )

    best_val = 0
    if os.path.isfile(best_model_checkpoint_loc):
        ckpt = torch.load(best_model_checkpoint_loc)
        best_val = ckpt['best_val']

    return latest_epoch + 1, best_val

def scale_features(std_scaler, X):
    p, t, din = X.shape[0], X.shape[1], X.shape[2]
    X_reshaped = X.reshape(p * t, din)
    X_scaled = std_scaler.transform(X_reshaped).reshape(p, t, din)
    return torch.tensor(X_scaled, dtype=torch.float32)


import math
class ValidationDatasetMetric:

    def __init__(self, model, validation_dataset, std_scaler, model_type = ModelType.DX_MODEL):
        self.model = model
        self.validation_dataset = validation_dataset
        self.std_scaler = std_scaler
        self.model_type = model_type
        batch_size = 1
        self.validation_dataset_dataloader = DataLoader(validation_dataset, batch_size=batch_size, pin_memory=True, in_order=False)

    def get_metrics(self):
        
        self.model.eval()
        self.model.to('cuda')

        loop = tqdm(self.validation_dataset_dataloader, total=len(self.validation_dataset_dataloader), desc=f"validating {self.model_type}")

        se_sum = 0
        instances = 0
        with torch.no_grad():
            for (x, y, predict_bool, num_output) in loop:

                x_ = scale_features(self.std_scaler, x.squeeze(0)).to('cuda')

                if self.model_type == ModelType.DX_MODEL:
                    y_ = y[0,:,:,0]
                else:
                    y_ = y[0,:,:,1]

                y_ = torch.as_tensor(y_, dtype=torch.float32, device='cuda')
                predict_bool_ = torch.tensor(predict_bool, device='cuda')
                num_output_ = int(num_output.item())

                output = self.model(x_, y_, predict_bool_, num_output_)

                se_sum+= output[1].item()
                instances+=y_.numel()
                
                del output

        mse = (se_sum / instances)
        rmse = math.sqrt(mse)
        return mse, rmse, se_sum, instances
    
    

    def evaluate(self):
        avg_mse, avg_rmse, se_sum, instances = self.get_metrics()
        return {'valid_mse': avg_mse , 'valid_rmse': avg_rmse, 'se_sum':se_sum, 'instances' : instances}





from torch.utils.data import DataLoader
import time
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import math

class Runner:

    def __init__(self, model, train_data, test_data, std_scaler, model_type = ModelType.DX_MODEL, lr=5e-5, wd=1e-4):
        

        batch_size = 1
        self.train_data_loader = DataLoader(train_data, batch_size=batch_size, pin_memory=True, in_order=False)
        self.model = model

        self.std_scaler = std_scaler

        self.validation_dataset_metric = ValidationDatasetMetric(self.model, test_data, std_scaler, model_type=model_type)

        self.model_type = model_type

        if self.model_type == ModelType.DX_MODEL:
            self.best_model_checkpoint_loc = BEST_DX_MODEL_CHECKPOINT
            self.checkpoint_dir = DX_CHECKPOINT_DIR
        else:
            self.best_model_checkpoint_loc = BEST_DY_MODEL_CHECKPOINT
            self.checkpoint_dir = DY_CHECKPOINT_DIR

        self.trainable_params = []
        for _, param in self.model.named_parameters():
            self.trainable_params.append(param)

        self.optimizer = torch.optim.AdamW(self.trainable_params, lr=lr, weight_decay=wd)



    def run(self):

        resume = False
        new_lr = 1e-5
        self.model.to('cuda')
        
        if resume:
            start_epoch, best_val = setup_state_from_checkpoint(self.model, self.optimizer, self.checkpoint_dir, self.best_model_checkpoint_loc)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
        else:
             start_epoch = 0
             best_val = math.inf

        end_epoch = start_epoch + 10
        validation_steps = 1500
        for epoch in range(start_epoch, end_epoch):

            loop = tqdm(self.train_data_loader, total=len(self.train_data_loader), desc=f"Epoch {epoch}")
            start = time.perf_counter()

            se_sum = 0
            instances = 0
            step = 0
            for (x, y, predict_bool, num_output) in loop:
                    
                x_ = scale_features(self.std_scaler, x.squeeze(0)).to('cuda')
                if self.model_type == ModelType.DX_MODEL:
                    y_ = y[0,:,:,0]
                else:
                    y_ = y[0,:,:,1]

                y_ = torch.as_tensor(y_, dtype=torch.float32, device='cuda')
                predict_bool_ = torch.tensor(predict_bool, device='cuda')
        
                num_output_ = int(num_output.item())
                
                output = self.model(x_, y_, predict_bool_, num_output_)
                
                se_sum+= output[1].item()
        
                instances+=y_.numel()
                
                mse_loss = output[0]
                mse_loss.backward()

                gradient_before_clipping = clip_grad_norm_(self.trainable_params, max_norm=1, foreach=True)
                self.optimizer.step()
                self.optimizer.zero_grad()

       
                loop.set_postfix({
                    'gradient': gradient_before_clipping.item(),
                    'step': step,
                    'loss': mse_loss.item(),
                    'learning_rates': '[' + ','.join(
                        [str(param_group['lr']) for param_group in self.optimizer.state_dict()['param_groups']]) + ']'
                    }
                )

                if (step + 1) % validation_steps == 0:
                    mse = se_sum / instances
                    rmse = math.sqrt(mse)
                    metrics = self.validation_dataset_metric.evaluate()
                    valid_mse = metrics['valid_mse']
                    valid_rmse = metrics['valid_rmse']
                    
                    score = valid_rmse
                    if score < best_val:
                        torch.save({'model_state': self.model.state_dict(), 'best_val': score}, self.best_model_checkpoint_loc)
                        best_val = score

                    end = time.perf_counter()
                    print(f'{{ step {step} train_avg_mse: {mse}, train_avg_rmse : {rmse} \n valid_avg_mse: {valid_mse} valid_avg_rmse:{valid_rmse} took {end - start:.3f} seconds}}')
                    checkpoint_state(epoch, self.model, self.optimizer, self.checkpoint_dir)
                    self.model.train()
                    se_sum = 0
                    instances = 0
                    

                step += 1

                del output, mse_loss, x_, y_, predict_bool_

                
            if (step % validation_steps) != 0:
                se_sum = 0
                instances = 0

def train(model, train_data, test_data, std_scaler, model_type = ModelType.DX_MODEL):
    runner = Runner(model, train_data, test_data, std_scaler, model_type, lr=1e-4, wd=1e-4)
    runner.run()

def evaluate(dx_model, dy_model, test_data, std_scaler, use_sub_sample_for_calculation=True):

    subset_input_groups = []
    subset_output_groups = []
    if use_sub_sample_for_calculation:
        for t in range(50):
            subset_input_groups.append(test_data.input_groups[t])
            subset_output_groups.append(test_data.output_groups[t])

        test_data = NFLDataset(subset_input_groups, subset_output_groups, test_data.nfl_feature_transformer)
    
    dx_checkpoint_pt = torch.load(BEST_DX_MODEL_CHECKPOINT, map_location='cpu')
    dx_model.load_state_dict(dx_checkpoint_pt['model_state'])
    
    dx_validation_dataset_metric = ValidationDatasetMetric(dx_model, test_data, std_scaler, model_type=ModelType.DX_MODEL)
    dx_metrics = dx_validation_dataset_metric.evaluate()

    dy_checkpoint_pt = torch.load(BEST_DY_MODEL_CHECKPOINT, map_location='cpu')
    dy_model.load_state_dict(dy_checkpoint_pt['model_state'])

    dy_validation_dataset_metric = ValidationDatasetMetric(dy_model, test_data, std_scaler, model_type=ModelType.DY_MODEL)
    dy_metrics = dy_validation_dataset_metric.evaluate()

    total_se_sum = dx_metrics['se_sum'] + dy_metrics['se_sum']
    total_instances = dx_metrics['instances'] + dy_metrics['instances']
    avg_mse = total_se_sum / total_instances if total_instances > 0 else 0
    avg_rmse = math.sqrt(avg_mse)

    avg_rmse_x = np.sqrt(dx_metrics['se_sum'] / dx_metrics['instances'])
    avg_rmse_y = np.sqrt(dy_metrics['se_sum'] / dy_metrics['instances'])
    print(f'model rmse on the test dataset on dx {avg_rmse_x}')
    print(f'model rmse on the test dataset on dy {avg_rmse_y}')
    print(f'model rmse on test dataset: {avg_rmse}')

def predict(dx_model, dy_model, test_data, std_scaler, use_sample_for_calculation=True):

    subset_input_groups = []
    subset_output_groups = []
    if use_sample_for_calculation:
        for t in range(20):
            subset_input_groups.append(test_data.input_groups[t])
            subset_output_groups.append(test_data.output_groups[t])

        test_data = NFLDataset(subset_input_groups, subset_output_groups, test_data.nfl_feature_transformer)
    
    dx_checkpoint_pt = torch.load(BEST_DX_MODEL_CHECKPOINT, map_location='cpu')
    dx_model.load_state_dict(dx_checkpoint_pt['model_state'])
    
    dy_checkpoint_pt = torch.load(BEST_DY_MODEL_CHECKPOINT, map_location='cpu')
    dy_model.load_state_dict(dy_checkpoint_pt['model_state'])

    dx_model.eval()
    dx_model.to('cuda')

    dy_model.eval()
    dy_model.to('cuda')

    preds = []
    with torch.no_grad():
        n = len(test_data)
        for t in tqdm(range(n), total=n, desc = 'dx_predict'):
            input_df = test_data.input_groups[t]
            output_df = test_data.output_groups[t]

            (x, _, predict_bool, num_output) = test_data[t]
            x_ = scale_features(std_scaler, x).to('cuda') # type: ignore
            
            predict_bool_ = torch.tensor(predict_bool, device='cuda')
            num_output_ = int(num_output.item())

            dx = dx_model(x_, None, predict_bool_, num_output_)

            dy = dy_model(x_, None, predict_bool_, num_output_)
            
            pred_play_sorted = input_df[input_df['player_to_predict']].sort_values(by=['nfl_id', 'frame_id'], ascending=[True, True])
            output_df_sorted = output_df.sort_values(by=['nfl_id', 'frame_id'], ascending=[True, True])
            predict_players_last_frame = (pred_play_sorted.groupby(['nfl_id'], as_index=False).last() )


            predict_players_output_frames = predict_players_last_frame.merge(output_df_sorted, on=['nfl_id'], how='left')
            
            pred_x =  -dx[:,0].cpu().numpy() + predict_players_output_frames['x_x'].values
            pred_y =  dy[:,0].cpu().numpy() + predict_players_output_frames['y_x'].values

            game_id = predict_players_output_frames['game_id_x'].values
            play_id = predict_players_output_frames['play_id_x'].values
            nfl_id =  predict_players_output_frames['nfl_id'].values
            frame_id = predict_players_output_frames['frame_id_y'].values
            player_position = predict_players_output_frames['player_position'].values
            player_role = predict_players_output_frames['player_role'].values
            x_last = predict_players_output_frames['x_x'].values
            y_last = predict_players_output_frames['y_x'].values
            play_direction = predict_players_output_frames['play_direction'].values

            actual_x = output_df_sorted['x'].values
            actual_y = output_df_sorted['y'].values

            columns = np.column_stack([game_id, play_id, nfl_id, frame_id, player_position, 
                          player_role,play_direction,x_last, y_last, actual_x, actual_y, pred_x, pred_y])
            
            preds.append(columns)
                       
    combined = np.concat(preds, axis=0)
    df = pd.DataFrame(combined, columns=['game_id', 'play_id', 'nfl_id','frame_id', 'player_position', 
                                       'player_role','play_direction', 'x_last', 'y_last', 
                                       'actual_x', 'actual_y', 'pred_x', 'pred_y'])
    
    df.to_csv('gru_test_data_results.csv', index=False)
    print('results published')
    



import joblib
from sklearn.preprocessing import StandardScaler

def save_joblib(train_data):
 
    def get_x(n):
        x,_,_,_ = train_data[n]
        p = x.shape[0] # type: ignore
        t = x.shape[1] # type: ignore
        din = x.shape[2] # type: ignore

        print(n)
        return x.reshape(p*t, din) # type: ignore
    
    percentage_of_sample = 0.02
    sample_n = int(len(train_data)*percentage_of_sample)
    sample_range = range(sample_n)
    with ThreadPoolExecutor(max_workers = 12) as executor:
            all_x = executor.map(get_x, sample_range)
            combined_x = np.concat(list(all_x),axis=0)
      
    std_scaler = StandardScaler()
    std_scaler.fit(combined_x)
    
    joblib.dump(std_scaler, JOBLIB_FILE_PATH)
    return std_scaler


def load_scaler():
    return joblib.load(JOBLIB_FILE_PATH)


print('loading dataframes...')
input_df, output_df = get_input_output_df()

print('splitting train test...')
nfl_feature_transformer = NFLFeatureTransformer()
train_data, test_data = split_train_test(input_df, output_df, nfl_feature_transformer, 0.02)


if not os.path.isfile(JOBLIB_FILE_PATH):
    print('fitting scaler and saving joblib...')
    std_scaler = save_joblib(train_data)
else:
    print('loading scaler from joblib...')
    std_scaler = load_scaler()


dx_model = SpatialTemporalLayer(d_model=128, nhead=8, spatial_encoder_layers=2, dropout=0.0, in_features=len(MODEL_INPUT_FEATURES))
dy_model = SpatialTemporalLayer(d_model=128, nhead=8, spatial_encoder_layers=2, dropout=0.0, in_features=len(MODEL_INPUT_FEATURES))

# print('training models...')
# train(dx_model, train_data, test_data,  std_scaler, model_type=ModelType.DX_MODEL)
# train(dy_model, train_data, test_data,  std_scaler, model_type=ModelType.DY_MODEL)

# print('predicting models...')
# predict(dx_model, dy_model, test_data, std_scaler, True)

print('evaluating models...')
evaluate(dx_model, dy_model, test_data, std_scaler, False)



