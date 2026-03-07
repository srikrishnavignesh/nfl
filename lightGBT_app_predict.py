
from lightGBT import get_test_df
import lightgbm as lgb
import numpy as np
import pandas as pd

dx_model = lgb.Booster(model_file='lightGBT_dx_model.txt')
dy_model = lgb.Booster(model_file='lightGBT_dy_model.txt')

BASE_COLS = ['game_id', 'play_id', 'player_to_predict', 'nfl_id', 'frame_id',
       'play_direction', 'absolute_yardline_number', 'player_name',
       'player_height', 'player_weight', 'player_birth_date',
       'player_position', 'player_side', 'player_role', 'x', 'y', 's', 'a',
       'dir', 'o', 'num_frames_output', 'ball_land_x', 'ball_land_y']

MANDATORY_COLS=['game_id', 'play_id', 'nfl_id']

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
            raise Exception(f'{c} in input contains all nan')
    
   
    X_df = get_test_df(df)

    dx_features = dx_model.feature_name()
    dy_features = dy_model.feature_name()

    pred_dx = dx_model.predict(X_df[dx_features])
    pred_dy = dy_model.predict(X_df[dy_features])
    
    game_id = X_df['game_id'].values
    play_id = X_df['play_id'].values
    nfl_id =  X_df['nfl_id'].values
    frame_id = X_df['frame_id'].values
    player_position = X_df['player_position'].values
    player_role = X_df['player_role'].values
    x_last = X_df['x_last'].values
    y_last = X_df['y_last'].values
    
    play_direction = X_df['play_direction'].values

    pred_x = pred_dx + x_last
    pred_y = pred_dy + y_last

    mask = (play_direction == 'left')

    if np.any(mask):
        pred_x[mask] =  120 - pred_x[mask]

    pred_x = np.clip(pred_x, 0.0, 120.0)
    pred_y = np.clip(pred_y, 0.0, 53.3)
  

    preds = np.column_stack([game_id, play_id, nfl_id, frame_id, player_position, 
                    player_role, play_direction, x_last, y_last, pred_x, pred_y])
    
        
    df = pd.DataFrame(preds, columns=['game_id', 'play_id', 'nfl_id','frame_id', 'player_position', 
                                    'player_role','play_direction', 
                                    'x_last', 'y_last', 'pred_x', 'pred_y'])
    
    return df.sort_values(by=['game_id', 'play_id', 'nfl_id', 'frame_id'])
    