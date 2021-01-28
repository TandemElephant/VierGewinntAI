from viergewinnt import *
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.activations import relu, tanh



if __name__ == '__main__':

    model1 = Sequential()
    model1.add(Conv2D(50, kernel_size=(4, 4), input_shape=(6, 7, 1), activation=relu))
    model1.add(Flatten())
    model1.add(Dense(128, activation=relu))
    model1.add(Dense(32, activation=relu))
    model1.add(Dense(1, activation=tanh))
    model1.compile(loss='mse')
    player1 = DeepAgent("LargerRobot1", value_model=model1, exploration_factor=0.2)
    # player1.load_model()

    model2 = Sequential()
    model2.add(Conv2D(50, kernel_size=(4, 4), input_shape=(6, 7, 1), activation=relu))
    model2.add(Flatten())
    model2.add(Dense(128, activation=relu))
    model2.add(Dense(32, activation=relu))
    model2.add(Dense(1, activation=tanh))
    model2.compile(loss='mse')
    player2 = DeepAgent("LargerRobot2", value_model=model2, exploration_factor=0.2)
    # player2.load_model()

    game = VierGewinnt(player1, player2)

    dummy1 = DeepAgent("Robot1")
    dummy1.load_model()
    dummy2 = DeepAgent("Robot2")
    dummy2.load_model()
    test_results_1, test_results_2 = game.play_games_for_learning(1000, test_player1=dummy1, test_player2=dummy2)

    player1.save_model('D:/Users/apist/PycharmProjects/VierGewinntAI')
    player2.save_model('D:/Users/apist/PycharmProjects/VierGewinntAI')

    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    # print(game.test_against_dummy(50, player1=player1))
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats()

    # player1 = DeepAgent('Robot1')
    # player2 = Player('Pityu')
    # player1.load_model()
    # game = VierGewinnt(player1, player2)
    # game.play_game()
