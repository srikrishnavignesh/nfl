# 📘 NFL Player Trajectory Prediction — Spatiotemporal Modeling (Kaggle Big Data Bowl 2026)

## 📌 Overview
This project is based on a Kaggle competition focused on predicting player positions in the NFL after the quarterback releases the ball.  
The challenge involves forecasting the next **N frames (on an average 11)** of movement for selected players using their past tracking data.

Player movement during a downfield pass is highly dynamic and uncertain — outcomes range from completions to interceptions. Understanding these trajectories helps the NFL analyze behavior of receivers, defenders, and passers during critical moments of a play.

---

## 🏈 Problem Description
During a pass play:

- The quarterback (QB) receives the snap.
- At the moment the ball is thrown, the prediction window begins.
- Tracking data provides positions and physical attributes for all players near the action.
- Not all 22 players are relevant — only those within the vicinity of the play (often 10–14 players).

### 🎯 Goal
Predict the **next N (20–40)** positions *(x, y)* for a subset of players (**P_pred ≤ P**) given:

- Their historical movement prior to the pass  
- Ball landing location  
- Player roles (Passer, Targeted Receiver, Defense, etc.)  
- Game- and play-level context  

---

## 📂 Input Features
Each frame includes tracking data before the pass is thrown:

- `game_id`: Game identifier, unique (numeric)  
- `play_id`: Play identifier, not unique across games (numeric)  
- `player_to_predict`: Whether or not the x/y prediction for this player will be scored (bool)  
- `nfl_id`: Player identification number, unique across players (numeric)  
- `frame_id`: Frame identifier for each play/type, starting at 1 for each game_id/play_id/file type (input or output) (numeric)  
- `play_direction`: Direction that the offense is moving (left or right)  
- `absolute_yardline_number`: Distance from end zone for possession team (numeric) (how far from the score zone)
- `player_name`: Player name (text)  
- `player_height`: Player height (ft-in)  
- `player_weight`: Player weight (lbs)  
- `player_birth_date`: Birth date (yyyy-mm-dd)  
- `player_position`: Player's position (role on the field)  
- `player_side`: Team player is on (Offense or Defense)  
- `player_role`: Role on the play (Defensive Coverage, Targeted Receiver, Passer, Other Route Runner)  
- `x`: Player position along the long axis of the field (0–120 yards)  
- `y`: Player position along the short axis of the field (0–53.3 yards)  
- `s`: Speed in yards/second (numeric)  
- `a`: Acceleration in yards/second² (numeric)  
- `o`: Orientation of player (degrees)  
- `dir`: Angle of player motion (degrees)  
- `num_frames_output`: Number of frames to predict in output data for the given game_id/play_id/nfl_id (numeric)  
- `ball_land_x`: Ball landing x-position (0–120 yards)  
- `ball_land_y`: Ball landing y-position (0–53.3 yards)

---

## 🛠 Feature Engineering

Feature engineering is performed using the `NFLFeatureTransformer` class, which normalizes numerical fields and adds several new features.

### 🔹 Angle-Based Features

- `sin_dir`, `cos_dir`  
- `sin_o`, `cos_o`  
- `angle_between_ball_land_and_player`  
- `angle_between_ball_land_and_player_orient`  
- `sin_angle_between_ball_land_and_player`  
- `cos_angle_between_ball_land_and_player`  
- `sin_angle_between_ball_land_and_player_orient`  
- `cos_angle_between_ball_land_and_player_orient`  

These help the model understand relative direction and orientation with respect to the ball.

### 🔹 Distance & Physics Features

- `velocity_x`, `velocity_y`  
- `acc_x`, `acc_y`  
- `distance_to_x`, `distance_to_y`  
- `distance_to_sideline`, `distance_to_endzone`  
- `distance_to_ball_land_x`, `distance_to_ball_land_y`  

These features capture how players move in relation to the field and ball landing location.

### 🔹 Temporal Change Features

- `time_left`  
- `required_speed`, `required_acceleration`  
- `change_in_x`, `change_in_y`  
- `change_in_speed`  
- `change_in_acceleration`  
- `change_in_o`  
- `change_in_dir`  

These model short-term motion dynamics and evolution of movement.

---

## 📁 Project Notebooks

- `gru_processing.ipynb`  

Each notebook represents a separate modeling approach to the same prediction task.



---

## 🔷 Model 2: GRU-Based Spatiotemporal Processing

### 🧱 Architecture

This approach combines:

1. **Spatial Transformer Encoder**  
   - Operates on `[T, P, din]`  
   - Captures relationships among players at a given time frame  

2. **GRU Layer**  
   - Operates on `[P, T, din]`  
   - Models temporal evolution of each player's trajectory  

Input is fed as:

- `[P, T, din]`

Where:

- `P`: total players under focus  
- `T`: total time steps  
- `din`: input feature dimension  

### 🎯 Prediction Strategy

- Uses **teacher forcing** to predict the next **N** frames for the `P_pred` players.  

### 📉 Loss Function

- **Mean Squared Error (MSE)**

### 📊 Performance

- **RMSE: 2.891 yards**

---

## 🧠 Key Insights

- Spatial encoders help understand player formations and local interactions.  
- Temporal modeling (speed, acceleration, orientation change) is crucial for accurate forecasting.  
- Physics- and geometry-based features (angles, distances, required speed) significantly boost performance.  
- GRU-based modeling gave more stable long-horizon predictions in this setup compared to pure transformer encoders.

---

## 🚀 Future Work
- Explore graph neural networks for player–player interaction modeling.  

---

## 🎉 Conclusion

This project applies deep feature engineering, transformers, and recurrent architectures to model complex NFL player movement in pass plays.  
The GRU-based model currently achieves better RMSE than the transformer-based variant, but both highlight the rich spatiotemporal structure in tracking data and open doors to more advanced architectures in future work.
