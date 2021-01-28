import numpy as np
from numpy_groupby import groupby_np
import random
from scipy.signal import convolve2d
from copy import deepcopy
import keras
from tqdm import tqdm

EMPTY = 0
NO_MOVE = -999

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
                move_values = self.calc_move_values(state, available_moves)
                result_move = random.choice(available_moves[move_values == np.max(move_values)])

        return result_move

    def make_move(self, state, learn=False):

        if learn is False:
            move = self.find_move(state)
        else:
            move = self.find_move(state, self.exp_factor)

        new_state = self.game.insert_piece(self.name, move, state)
        return new_state

    def calc_state_values(self, states):
        return self.value_model.predict(states.reshape(*states.shape, 1)).ravel()

    def calc_move_values(self, state, moves):
        # calculate all possible states after #self.iteration number of self and opp moves,
        # remember which moves led to which moves in self_move_idx and opp_move_idx
        # calculate values for all states, then minimize/maximize according to remembered move indices

        self_move_idx = []
        opp_move_idx = []

        moves_and_temp_states = np.array([['', f'{move}', self.game.insert_piece(self.name, move, state)]
                                          for move in moves],
                                         dtype='object')

        moves_and_temp_states = np.array([[prev_move, f'{prev_move}{move if move != NO_MOVE else ""}',
                                           self.game.insert_piece(self.name, move, temp_state)
                                           ]
                                          for prev_move, temp_state in moves_and_temp_states[:, 1:]
                                          for move in self.game.get_available_moves(temp_state)],
                                         dtype='object')
        opp_move_idx.append(moves_and_temp_states[:, 0])

        for i in range(self.iteration-1):
            moves_and_temp_states = np.array([[prev_move, f'{prev_move}{move if move != NO_MOVE else ""}',
                                               self.game.insert_piece(self.name, move, temp_state)
                                               ]
                                              for prev_move, temp_state in moves_and_temp_states[:, 1:]
                                              for move in self.game.get_available_moves(temp_state)],
                                             dtype='object')
            self_move_idx.append(moves_and_temp_states[:, 0])

            moves_and_temp_states = np.array([[prev_move, f'{prev_move}{move if move != NO_MOVE else ""}',
                                               self.game.insert_piece(self.name, move, temp_state)
                                               ]
                                              for prev_move, temp_state in moves_and_temp_states[:, 1:]
                                              for move in self.game.get_available_moves(temp_state)],
                                             dtype='object')
            opp_move_idx.append(moves_and_temp_states[:, 0])

        final_states = np.array([*moves_and_temp_states[:, 2]])
        state_values = self.calc_state_values(final_states)

        # check state winners
        winners = [self.game.check_winner(temp_state) for temp_state in final_states]
        state_values = np.array([value if winner is None else self.calc_reward(winner)
                                 for value, winner in zip(state_values, winners)])

        state_values = groupby_np(state_values, opp_move_idx[-1], uf=np.minimum)
        for i in range(self.iteration-2, -1, -1):
            state_values = groupby_np(state_values, self_move_idx[i], uf=np.maximum)
            state_values = groupby_np(state_values, opp_move_idx[i], uf=np.minimum)

        return state_values

    def learn_from_game(self, game_history, winner):

        self_states = [state for name, state in game_history if name == self.name]

        # learn reward for last state
        reward = self.calc_reward(winner)
        last_state_value = self.calc_state_values(self_states[-1].reshape(1, *self_states[-1].shape))
        value_diff = reward - last_state_value
        target = last_state_value + self.learning_rate * value_diff
        self.value_model.fit(self_states[-1].reshape(1, *self_states[-1].shape, 1),
                             np.array(target),
                             epochs=self.epochs, verbose=0)

        # learn values of previous states
        for state, prev_state in zip(self_states[len(self_states)-1:0:-1], self_states[len(self_states)-2::-1]):
            state_value = self.calc_state_values(state.reshape(1, *state.shape))
            prev_state_value = self.calc_state_values(prev_state.reshape(1, *prev_state.shape))
            value_diff = state_value - prev_state_value
            target_for_prev_state = prev_state_value + self.learning_rate * value_diff
            self.value_model.fit(prev_state.reshape(1, *prev_state.shape, 1),
                                 np.array(target_for_prev_state),
                                 epochs=self.epochs, verbose=0)

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
        self.state = np.zeros((6, 7))
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

        if column == NO_MOVE:
            return state

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

        if self.check_winner(state) is None:
            return np.arange(7)[np.sum(state == EMPTY, axis=0) > 0]
        else:
            return np.array([NO_MOVE])
