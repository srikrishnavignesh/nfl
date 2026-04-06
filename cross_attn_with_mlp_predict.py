from cross_attn_with_mlp import NFLFeatureTransformer, SpatialTemporalLayer, scale_features, MODEL_INPUT_FEATURES
from torch.utils.data import Dataset
import os
import numpy as np
import joblib
import torch
import pandas as pd


class NFLDataset(Dataset):
    def __init__(self, input_groups, nfl_feature_transformer):
        self.nfl_feature_transformer = nfl_feature_transformer
        self.input_groups = input_groups
        
        
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

        play_direction = input_df.iloc[0]['play_direction']

        input_df = self.nfl_feature_transformer.reflect_input_coordinates(input_df, play_direction)

        given_frames = input_df['frame_id'].max()
        output_frames = input_df['num_frames_output'].iloc[0]

        total_frames = given_frames + output_frames

        min_frame_start = max(given_frames - 10, 1)
        
        prev_frame = None
        x = None
        for f in range(min_frame_start, given_frames+1):
            
            grouped_frame = input_df[input_df['frame_id'] == f].sort_values(by=['nfl_id'], ascending=[True]).reset_index(drop=True)

            prev_frame = grouped_frame if prev_frame is None else prev_frame
            transformed_input_df = self.nfl_feature_transformer.transform_X(grouped_frame, prev_frame, total_frames)
            input_array = self._get_input_array(transformed_input_df)
            x = input_array if x is None else np.concat((x, input_array), axis=1)
            prev_frame = grouped_frame
        
        
        num_output = prev_frame['num_frames_output'].iloc[0] # type: ignore
        players_to_predict = prev_frame['player_to_predict'].values.tolist() # type: ignore

        return x, players_to_predict, num_output
    
    


JOBLIB_FILE_PATH = f'cross_attn_with_mlp_scaler.joblib'
BEST_DX_MODEL_CHECKPOINT = f'cross_attn_best_weight_dx.pt'
BEST_DY_MODEL_CHECKPOINT = f'cross_attn_best_weight_dy.pt'

BASE_COLS = ['game_id', 'play_id', 'player_to_predict', 'nfl_id', 'frame_id',
       'play_direction', 'absolute_yardline_number', 'player_name',
       'player_height', 'player_weight', 'player_birth_date',
       'player_position', 'player_side', 'player_role', 'x', 'y', 's', 'a',
       'dir', 'o', 'num_frames_output', 'ball_land_x', 'ball_land_y']

MANDATORY_COLS=['game_id', 'play_id', 'nfl_id']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


nfl_feature_transformer = NFLFeatureTransformer()

std_scaler = joblib.load(JOBLIB_FILE_PATH)


dx_model = SpatialTemporalLayer(d_model=128, nhead=8, spatial_encoder_layers=2, dropout=0.0, in_features=len(MODEL_INPUT_FEATURES))
dy_model = SpatialTemporalLayer(d_model=128, nhead=8, spatial_encoder_layers=2, dropout=0.0, in_features=len(MODEL_INPUT_FEATURES))

dx_checkpoint_pt = torch.load(BEST_DX_MODEL_CHECKPOINT, map_location='cpu')
dx_model.load_state_dict(dx_checkpoint_pt['model_state'], strict=False)

dy_checkpoint_pt = torch.load(BEST_DY_MODEL_CHECKPOINT, map_location='cpu')
dy_model.load_state_dict(dy_checkpoint_pt['model_state'], strict=False)


dx_model.eval()
dx_model.to(device)

dy_model.eval()
dy_model.to(device)

def predict(df):

    given_input_cols = set(df.columns)
    for c in MANDATORY_COLS:
        if c not in given_input_cols:
            raise Exception(f'{c} is missing in input')
        elif df[c].isna().any():
            raise Exception(f'{c} in input contains nan')
        
    
    for c in BASE_COLS:
        if c not in df:
            raise Exception(f'{c} is not there in input')
        if df[c].isna().all():
            raise Exception(f'{c} in input contains nan')
    
    input_groups = [group for _, group in df.groupby(by=['game_id', 'play_id'], sort=True)]
    test_data = NFLDataset(input_groups, nfl_feature_transformer)
    
    
    n = len(test_data)

    preds = []
    with torch.no_grad():
        for i in range(n):

            df = test_data.input_groups[i]

            (x, predict_bool, num_output) = test_data[i]
            x_ = scale_features(std_scaler, x).to(device)
            
            predict_bool_ = torch.tensor(predict_bool, device=device)
            num_output_ = int(num_output.item())
            
            dx = dx_model(x_, None, predict_bool_, num_output_)

            dy = dy_model(x_, None, predict_bool_, num_output_)
            
            predict_players = df[df['player_to_predict']]
            
            #this makes sure that last available data is fetched
            predict_players_last_frame = predict_players.sort_values(by=['frame_id']).groupby(by=['nfl_id'], 
                                                                                              as_index=False, sort=False).last()
            
            num_frames_output = predict_players['num_frames_output'].iloc[0]

            output_frames_n = list(range(1, num_frames_output+1))

            output_frames = pd.DataFrame(output_frames_n, columns=['frame_id'])


            predict_players_output_frames = (predict_players_last_frame.merge(output_frames, how='cross')
                                                    .sort_values(by=['nfl_id', 'frame_id_y']).reset_index(drop=True) 
                                                )
            
            f = 1
            if predict_players_output_frames['play_direction'].iloc[0] == 'left':
                f = -1

            pred_x =  predict_players_output_frames['x'].values + f*dx[:,0].cpu().numpy()
            pred_y =  predict_players_output_frames['y'].values + dy[:,0].cpu().numpy() 

            game_id = predict_players_output_frames['game_id'].values
            play_id = predict_players_output_frames['play_id'].values
            nfl_id =  predict_players_output_frames['nfl_id'].values
            frame_id = predict_players_output_frames['frame_id_y'].values
            player_position = predict_players_output_frames['player_position'].values
            player_role = predict_players_output_frames['player_role'].values
            x_last = predict_players_output_frames['x'].values
            y_last = predict_players_output_frames['y'].values
            play_direction = predict_players_output_frames['play_direction'].values

            pred_x = np.clip(pred_x, 0.0, 120.0)
            pred_y = np.clip(pred_y, 0.0, 53.3)


            columns = np.column_stack([game_id, play_id, nfl_id, frame_id, player_position, 
                          player_role,play_direction,x_last, y_last, pred_x, pred_y])
            
            preds.append(columns)
    
    combined = np.concat(preds, axis=0)
    df = pd.DataFrame(combined, columns=['game_id', 'play_id', 'nfl_id','frame_id', 'player_position', 
                                       'player_role','play_direction', 'x_last', 'y_last', 'pred_x', 'pred_y'])
    
    return df.sort_values(by=['game_id', 'play_id', 'nfl_id', 'frame_id'])




