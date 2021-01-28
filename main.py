from viergewinnt import *
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.activations import relu, sigmoid


if __name__ == '__main__':

    model1 = Sequential()
    model1.add(Conv2D(20, kernel_size=(4, 4), input_shape=(6, 7, 1), activation=relu))
    model1.add(Flatten())
    model1.add(Dense(10, activation=relu))
    model1.add(Dense(1, activation=sigmoid))
    model1.compile(loss='mse')
    player1 = DeepAgent("Robot1", value_model=model1, exploration_factor=1, iteration=2)

    model2 = Sequential()
    model2.add(Conv2D(20, kernel_size=(4, 4), input_shape=(6, 7, 1), activation=relu))
    model2.add(Flatten())
    model2.add(Dense(64, activation=relu))
    model2.add(Dense(16, activation=relu))
    model2.add(Dense(1, activation=sigmoid))
    model2.compile(loss='mse')
    player2 = DeepAgent("Robot2", value_model=model2, exploration_factor=1)

    game = VierGewinnt(player1, player2)

    test_results_1, test_results_2 = game.play_games_for_learning(1000)
    player1.save_model('D:/Users/apist/PycharmProjects/VierGewinntAI')
    player2.save_model('D:/Users/apist/PycharmProjects/VierGewinntAI')

    print("Player1 win rate against dummy:")
    print(test_results_1)
    print("\n")

    print("Player2 win rate against dummy:")
    print(test_results_2)
    print("\n")

    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    # print(game.test_against_dummy(50, player1=player1))
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats()

# # play against AI
# player1 = Player()
# game = VierGewinnt(player1, player2)
# game.play_game()
