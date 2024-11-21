import chess
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Encode board state into a fixed-size array
def encode_board(board):
    encoded = np.zeros(64, dtype=int)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        encoded[square] = piece.piece_type
    return encoded

# Generate dummy training data (for illustration purposes)
def generate_training_data(num_samples):
    X = []
    y = []
    for _ in range(num_samples):
        board = chess.Board()
        board_array = encode_board(board)
        X.append(board_array)
        # Random target values
        y.append(np.random.rand())
    return np.array(X), np.array(y)

# Create a simple model
def create_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(64,)),  # 64 squares on the chess board
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Generate dummy data
X_train, y_train = generate_training_data(1000)

# Create and train the model
model = create_model()
model.fit(X_train, y_train, epochs=10)

# Save the model
model.save('chess_model.h5')

