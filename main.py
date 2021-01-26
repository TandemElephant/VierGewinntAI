from viergewinnt import *
from keras.models import Sequential
from keras.layers import Conv2D, Dense
from keras.activations import relu, sigmoid


model1 = Sequential()
model1.add(Conv2D(10, kernel_size=(4, 4), input_shape=(6, 7, 1), activation=relu))
model1.add(Dense(10, activation=relu))
model1.add(Dense(1, activation=sigmoid))
model1.compile(loss='mse')

player1 = DeepAgent(value_model=model1, exploration_factor=1)


model2 = Sequential()
model2.add(Conv2D(10, kernel_size=(4, 4), input_shape=(6, 7, 1), activation=relu))
model2.add(Dense(64, activation=relu))
model2.add(Dense(16, activation=relu))
model2.add(Dense(1, activation=sigmoid))
model2.compile(loss='mse')

player2 = DeepAgent(value_model=model2, exploration_factor=1)

game = VierGewinnt(player1, player2)
game.play_games_for_learning(200)

# play against AI
player1 = Player()
game = VierGewinnt(player1, player2)
game.play_game()
