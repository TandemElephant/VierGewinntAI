import numpy as np
import random
from scipy.signal import convolve2d
from copy import deepcopy
import keras
from tqdm import tqdm

EMPTY = 0


class FullBoardException(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class FullColumnException(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class Player:
    def __init__(self, name):
        self.game = Game()
        self.is_human = True
        self.name = name

    def make_move(self, state, learn=False):
        available_moves = self.game.get_available_moves(state)
        move = int(input(f'Choose a move {available_moves}: '))
        new_state = self.game.insert_piece(self.name, move)
        return new_state


class DeepAgent(Player):
    def __init__(self, name, learning_rate=0.1, exploration_factor=0.2, iteration=1, train_epoch_per_move=10,
                 value_model=keras.models.Sequential()):
        super().__init__(name)
        self.learning_rate = learning_rate
        self.exp_factor = exploration_factor
        self.is_human = False
        self.iteration = iteration
        self.value_model = value_model
        self.epochs = train_epoch_per_move

    def find_move(self, state, exp_factor=0.0):

        available_moves = self.game.get_available_moves()

        # no available move
        if len(available_moves) == 0:
            raise FullBoardException("Cannot make move, board is full.")

        # one available move
        elif len(available_moves) == 1:
            result_move = available_moves[0]

        # more than one available move
        else:

            # make random exploration move
            if random.random() < exp_factor:
                result_move = random.choice(available_moves)

            # make optimal move
            else:
                optimal_move_value = -np.inf
                optimal_moves = []
                for move in available_moves:
                    move_value = self.calc_move_values(state, move)
                    if move_value > optimal_move_value:
                        optimal_moves = [move]
                        optimal_move_value = move_value
                    elif move_value == optimal_move_value:
                        optimal_moves.append(move)

                result_move = random.choice(optimal_moves)

        return result_move

    def make_move(self, state, learn=False):

        if learn is False:
            move = self.find_move(state)
        else:
            move = self.find_move(state, self.exp_factor)

        new_state = self.game.insert_piece(self.name, move, state)
        return new_state

    def calc_state_value(self, state):
        return self.value_model.predict(state.reshape(1, *state.shape, 1))

    def calc_move_values(self, state, moves, iteration=None):
        if iteration is None:
            iteration = self.iteration

        temp_states = [self.game.insert_piece(self.name, move, state) for move in moves]

        # check if won
        winner = self.game.check_winner(temp_state)
        if winner is not None:
            return self.calc_reward(winner)

        available_opp_moves = self.game.get_available_moves(temp_state)
        opp_move_values = []

        for opp_move in available_opp_moves:
            temp_state_after_opp = self.game.insert_piece(self.game.player2.name, opp_move, temp_state)
            if iteration == 1:
                opp_move_values.append(self.calc_state_value(temp_state_after_opp))
            else:
                available_self_moves = self.game.get_available_moves(temp_state_after_opp)
                if len(available_self_moves) <= 1:
                    opp_move_values.append(self.calc_state_value(temp_state_after_opp))
                else:
                    self_move_values = []
                    for self_move in available_self_moves:
                        self_move_values.append(self.calc_move_values(temp_state_after_opp, self_move, iteration - 1))
                    opp_move_values.append(np.max(self_move_values))

        return np.min(opp_move_values)

    def learn_from_game(self, game_history, winner):

        self_states = [state for name, state in game_history if name == self.name]

        # learn reward for last state
        reward = self.calc_reward(winner)
        last_state_value = self.calc_state_value(self_states[-1])
        value_diff = reward - last_state_value
        target = last_state_value + self.learning_rate * value_diff
        self.value_model.fit(self_states[-1].reshape(1, *self_states[-1].shape, 1),
                             np.array(target),
                             epochs=self.epochs, verbose=0)

        # learn values of previous states
        for state, prev_state in zip(self_states[len(self_states)-1:0:-1], self_states[len(self_states)-2::-1]):
            state_value = self.calc_state_value(state)
            prev_state_value = self.calc_state_value(prev_state)
            value_diff = state_value - prev_state_value
            target_for_prev_state = prev_state_value + self.learning_rate * value_diff
            self.value_model.fit(prev_state.reshape(1, *prev_state.shape, 1),
                                 np.array(target_for_prev_state),
                                 epochs=self.epochs, verbose=0)

    def calc_target(self, state, winner):

        prev_value = self.calc_state_value(self.prev_state)

        if winner is None:
            value_diff = self.calc_state_value(state) - prev_value
        else:
            value_diff = self.calc_reward(winner) - prev_value

        target = prev_value + self.learning_rate * value_diff
        return target

    def calc_reward(self, winner):
        if winner == self.name:
            return 1
        elif winner is None:
            return 0
        elif winner is False:  # draw
            return 0
        else:  # loss
            return -1

    def save_model(self, path='./'):
        file_path = f'{path}/deepagent_{self.name}.h5'
        self.value_model.save(file_path)

class Game:
    def __init__(self):
        pass

    def init_game(self):
        pass

    def play_game(self):
        pass

    def play_games_for_learning(self, n_games: int):
        pass

    def get_available_moves(self, state=None):
        return []

    def check_winner(self, state=None):
        return None

    def insert_piece(self, player_name, move, state=None):
        return state

class VierGewinnt(Game):
    def __init__(self, player1: Player, player2: Player):

        super().__init__()

        self.markers = ['X', 'O']
        self.vals = [1, -1]
        self.marker2val = {marker: val for marker, val in zip(self.markers, self.vals)}
        self.val2marker = {val: marker for marker, val in zip(self.markers, self.vals)}

        # assign players
        self.name2val = {}

        self.player1 = player1
        self.player1.game = self
        self.name2val[player1.name] = self.vals[0]
        self.player2 = player2
        self.player2.game = self
        self.name2val[player2.name] = self.vals[1]

        self.val2name = {val: name for name, val in self.name2val.items()}

        # init game
        self.state = np.array((6, 7))
        self.winner = None
        self.turn_player = self.player1
        self.game_history = []

    def init_game(self):

        self.state = np.zeros((6, 7))
        self.winner = None
        self.turn_player = self.player1
        self.game_history = []
        
    def play_game(self):
        self.init_game()

        while self.winner is None:

            if self.turn_player.is_human:
                self.print_game()
                print(f"{self.turn_player.name}'s turn.")

            self.state = self.play_move()

            self.winner = self.check_winner()
            if self.winner is not None:
                break

        # record winning position
        self.game_history.append((self.turn_player.name, self.state))

        if self.player1.is_human or self.player2.is_human:
            self.print_game()
            print(f"The winner is: {self.winner}")

        return self.winner

    def play_games_for_learning(self, n_games: int):

        test_results_1 = []
        test_results_2 = []

        for i in tqdm(range(n_games)):

            self.init_game()

            while self.winner is None:
                self.state = self.play_move(learn=True)
                self.winner = self.check_winner()

                if self.winner is not None:
                    break

                self.state = self.play_move(learn=True)
                self.winner = self.check_winner()

            self.game_history.append((self.turn_player.name, self.state))

            self.player1.learn_from_game(self.game_history, self.winner)
            self.player2.learn_from_game(self.game_history, self.winner)

            # test on dummies
            if (i+1) % 100 == 0:
                test_results_1.append(self.test_against_dummy(50, player1=self.player1))
                test_results_2.append(self.test_against_dummy(50, player2=self.player2))

        return test_results_1, test_results_2

    def test_against_dummy(self, n_games, player1=None, player2=None):

        dummy = DeepAgent("Dummy", exploration_factor=1)
        test_player = DeepAgent("Test")

        if player1 is not None and player2 is None:
            test_player.value_model = player1.value_model
            test_game = VierGewinnt(test_player, dummy)
        elif player1 is None and player2 is not None:
            test_player.value_model = player2.value_model
            test_game = VierGewinnt(dummy, test_player)
        else:
            raise ValueError

        score = 0
        for i in range(n_games):
            winner = test_game.play_game()
            score += float(winner == test_player.name) / n_games

        return score

    def check_winner(self, state=None):
        if state is None:
            state = self.state

        winner_positions = [
            np.array([[1],
                      [1],
                      [1],
                      [1]]),
            np.array([[1, 1, 1, 1]]),
            np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]),
            np.array([[0, 0, 0, 1],
                      [0, 0, 1, 0],
                      [0, 1, 0, 0],
                      [1, 0, 0, 0]]),
        ]

        winner = None
        for win_pos in winner_positions:
            convolved_state = convolve2d(state, win_pos, mode='valid')

            if (convolved_state == 4).any():
                winner = self.player1.name
                break
            elif (convolved_state == -4).any():
                winner = self.player2.name
                break
            elif (convolved_state != EMPTY).all():
                winner = False
                break

        return winner

    def play_move(self, learn=False):

        self.game_history.append((self.turn_player.name, self.state))

        new_state = self.turn_player.make_move(self.state, learn)

        self.next_player()
        return new_state

    def next_player(self):
        if self.turn_player.name == self.player1.name:
            self.turn_player = self.player2
        else:
            self.turn_player = self.player1

    def print_game(self):
        print(self.state)

    def insert_piece(self, player_name, column, state=None):
        if state is None:
            state = self.state

        if np.sum(state[:, column] == EMPTY) == 0:
            raise FullColumnException(f"Column {column} is full, cannot insert piece.")

        # find last available space in chosen column and fill it
        row = np.sum(state[:, column] == EMPTY) - 1
        new_state = deepcopy(state)
        new_state[row, column] = self.name2val[player_name]
        return new_state

    def get_available_moves(self, state=None):
        if state is None:
            state = self.state

        return np.arange(7)[np.sum(state == EMPTY, axis=0) > 0]