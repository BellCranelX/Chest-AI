import chess
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import urllib.request
from IPython.display import display, clear_output
import ipywidgets as widgets

# Initialize the chess board
board = chess.Board()

# Load the trained neural network model
model = tf.keras.models.load_model('chess_model.h5')

def encode_board(board):
    encoded = np.zeros(64, dtype=int)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        encoded[square] = piece.piece_type
    return encoded

def evaluate_board(board):
    board_array = encode_board(board)
    board_array = np.expand_dims(board_array, axis=0)
    return board_array

def choose_move(board, model):
    legal_moves = list(board.legal_moves)
    best_move = None
    best_value = -np.inf

    for move in legal_moves:
        board.push(move)
        board_value = model.predict(evaluate_board(board))[0][0]
        board.pop()

        if board_value > best_value:
            best_value = board_value
            best_move = move

    return best_move

# URLs of PNG piece images
piece_image_urls = {
    'P': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Chess_plt45.svg/45px-Chess_plt45.svg.png',
    'R': 'https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/Chess_rlt45.svg/45px-Chess_rlt45.svg.png',
    'N': 'https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/Chess_nlt45.svg/45px-Chess_nlt45.svg.png',
    'B': 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/Chess_blt45.svg/45px-Chess_blt45.svg.png',
    'Q': 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Chess_qlt45.svg/45px-Chess_qlt45.svg.png',
    'K': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/42/Chess_klt45.svg/45px-Chess_klt45.svg.png',
    'p': 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/Chess_pdt45.svg/45px-Chess_pdt45.svg.png',
    'r': 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Chess_rdt45.svg/45px-Chess_rdt45.svg.png',
    'n': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/Chess_ndt45.svg/45px-Chess_ndt45.svg.png',
    'b': 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/Chess_bdt45.svg/45px-Chess_bdt45.svg.png',
    'q': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Chess_qdt45.svg/45px-Chess_qdt45.svg.png',
    'k': 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/f0/Chess_kdt45.svg/45px-Chess_kdt45.svg.png'
}

# Download and open piece images
piece_images = {}
for piece, url in piece_image_urls.items():
    local_path = f"{piece}.png"
    urllib.request.urlretrieve(url, local_path)
    image = Image.open(local_path).convert('RGBA')
    piece_images[piece] = image.resize((45, 45))

def draw_board(board):
    # Create a new image with white background
    board_image = Image.new('RGB', (360, 360), 'white')
    draw = ImageDraw.Draw(board_image)

    # Draw the chessboard squares
    square_size = 45
    for i in range(8):
        for j in range(8):
            color = (209, 139, 71) if (i + j) % 2 == 0 else (255, 206, 158)
            draw.rectangle([i * square_size, j * square_size, (i+1) * square_size, (j+1) * square_size], fill=color)

    # Draw the pieces
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        piece_image = piece_images[str(piece)]
        x = chess.square_file(square) * square_size
        y = (7 - chess.square_rank(square)) * square_size
        board_image.paste(piece_image, (x, y), piece_image)

    return board_image

def on_submit_move(text_input):
    global board
    move = text_input.value
    try:
        board.push_san(move)
    except ValueError:
        print("Invalid move. Try again.")
        return

    if board.is_game_over():
        print("Game over!")
        print(board.result())
        return

    # AI (black) move
    ai_move = choose_move(board, model)
    if ai_move is None:
        print("No valid moves available. Game over.")
        return
    board.push(ai_move)
    print(f"AI played: {ai_move}")

    if board.is_game_over():
        print("Game over!")
        print(board.result())
        return

    display(draw_board(board))
    text_input.value = ""

def play_game():
    display(draw_board(board))
    text_input = widgets.Text(placeholder="Enter your move")
    text_input.on_submit(on_submit_move)
    display(text_input)

if __name__ == "__main__":
    play_game()