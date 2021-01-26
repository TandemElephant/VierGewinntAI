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
    def __init__(self):
        self.game = Game()
        self.is_human = True
        self.tag = None

    def make_move(self, state):
        if self.tag is None:
            print('Please put the player in a game before making a move.')
            return state
        else:
            column = int(input('Choose a column (0-6): '))
            new_state = self.game.insert_piece(self.tag, column)
            return new_state


class DeepAgent(Player):
    def __init__(self, learning_rate=0.1, exploration_factor=0.2, iteration=1, train_epoch_per_move=10,
                 value_model=keras.models.Sequential()):
        super().__init__()
        self.learning_rate = learning_rate
        self.exp_factor = exploration_factor
        self.is_human = False
        self.iteration = iteration
        self.value_model = value_model
        self.epochs = train_epoch_per_move
        self.prev_state = None

    def make_move(self, state, exp_factor=0.0):

        available_moves = self.game.get_available_moves()

        # no available move
        if len(available_moves) == 0:
            raise FullBoardException("Cannot make move, board is full.")

        # one available move
        elif len(available_moves) == 1:
            new_state = self.game.insert_piece(self.tag, available_moves[0],  state)

        # more than one available move
        else:

            # make random exploration move
            if random.random() < exp_factor:
                col = random.choice(available_moves)
                new_state = self.game.insert_piece(self.tag, col,  state)

            # make optimal move
            else:
                optimal_move_value = -np.inf
                optimal_moves = []
                for col in available_moves:
                    move_value = self.calc_move_value(state, col)
                    if move_value > optimal_move_value:
                        optimal_moves = [col]
                        optimal_move_value = move_value
                    elif move_value == optimal_move_value:
                        optimal_moves.append(col)

                optimal_move = random.choice(optimal_moves)
                new_state = self.game.insert_piece(self.tag, optimal_move, state)

        return new_state

    def make_move_and_learn(self, state, winner):

        self.learn_state(state, winner)

        if winner is None:
            new_state = self.make_move(state, exp_factor=self.exp_factor)
        else:
            new_state = state

        return new_state

    def calc_state_value(self, state):
        return self.value_model.predict(state.reshape(1, *state.shape, 1))

    def calc_move_value(self, state, move, iteration=None):
        if iteration is None:
            iteration = self.iteration

        temp_state = self.game.insert_piece(self.tag, move, state)
        available_opp_moves = self.game.get_available_moves(temp_state)
        opp_move_values = []

        for opp_move in available_opp_moves:
            temp_state_after_opp = self.game.insert_piece(self.game.player2.tag, opp_move, temp_state)
            if iteration == 1:
                opp_move_values.append(self.calc_state_value(temp_state_after_opp))
            else:
                available_self_moves = self.game.get_available_moves(temp_state_after_opp)
                if len(available_self_moves) <= 1:
                    opp_move_values.append(self.calc_state_value(temp_state_after_opp))
                else:
                    self_move_values = []
                    for self_move in available_self_moves:
                        self_move_values.append(self.calc_move_value(temp_state_after_opp, self_move, iteration - 1))
                    opp_move_values.append(np.max(self_move_values))

        return np.min(opp_move_values)

    def learn_state(self, state, winner):

        if self.prev_state is None:
            pass
        else:
            target_for_prev_state = self.calc_target(state, winner)
            self.value_model.fit(self.prev_state.reshape(1, *self.prev_state.shape, 1),
                                 np.array(target_for_prev_state),
                                 epochs=self.epochs, verbose=0)

        self.prev_state = state

    def calc_target(self, state, winner):

        prev_value = self.calc_state_value(self.prev_state)

        if winner is None:
            value_diff = self.calc_state_value(state) - prev_value
        else:
            value_diff = self.calc_reward(winner) - prev_value

        target = prev_value + self.learning_rate * value_diff
        return target

    def calc_reward(self, winner):
        if winner == self.tag:
            return 1
        elif winner is None:
            return 0
        elif winner is False:  # draw
            return 0
        else:  # loss
            return -1


class Game:
    def __init__(self):
        pass

    def init_game(self):
        pass

    def play_game(self):
        pass

    def play_games_for_learning(self):
        pass

    def get_available_moves(self, state=None):
        return []

    def check_winner(self, state=None):
        return None

    def insert_piece(self, tag, move, state=None):
        return state

class VierGewinnt(Game):
    def __init__(self, player1: Player, player2: Player):

        self.tags = ['X','O']
        self.vals = [1, -1]
        self.tag2val = {tag: val for tag, val in zip(self.tags, self.vals)}
        self.val2tag = {val: tag for tag, val in zip(self.tags, self.vals)}

        self.player1 = player1
        self.player1.game = self
        self.player1.tag = self.tags[0]
        self.player2 = player2
        self.player2.game = self
        self.player2.tag = self.tags[1]

        # init game
        self.state = np.zeros((6, 7))
        self.winner = None
        self.turn = self.tags[0]
        self.turn_player = self.player1

    def init_game(self):
        self.state = np.zeros((6, 7))
        self.winner = None
        self.turn = 'X'
        self.turn_player = self.player1
        
    def play_game(self):
        self.init_game()

        while self.winner is None:

            if self.turn_player.is_human:
                self.print_game()
                print(f"{self.turn_player.tag}'s turn.")

            self.state = self.play_move()

            self.winner = self.check_winner()
            if self.winner is not None:
                break

        self.print_game()
        print(f"The winner is: {self.winner}")

    def play_games_for_learning(self, n_games):

        for i in tqdm(range(n_games)):

            while self.winner is None:
                self.state = self.play_move(learn=True)
                self.winner = self.check_winner()

                if self.winner is not None:
                    break

                self.state = self.play_move(learn=True)
                self.winner = self.check_winner()

            # update loser's state
            self.state = self.play_move(learn=True)
            self.state = self.play_move(learn=True)
            # update winner's state
            self.state = self.play_move(learn=True)
            self.state = self.play_move(learn=True)

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
                winner = self.player1.tag
                break
            elif (convolved_state == -4).any():
                winner = self.player2.tag
                break
            elif (convolved_state != EMPTY).all():
                winner = False
                break

        return winner

    def play_move(self, learn=False):

        if learn is True:
            new_state = self.turn_player.make_move_and_learn(self.state, self.winner)
        else:
            new_state = self.turn_player.make_move(self.state)

        self.next_player()
        return new_state

    def next_player(self):
        if self.turn == self.player1.tag:
            self.turn = self.player2.tag
            self.turn_player = self.player2
        else:
            self.turn = self.player1.tag
            self.turn_player = self.player1

    def print_game(self):
        print(self.state)

    def insert_piece(self, tag, column, state=None):
        if state is None:
            state = self.state

        if np.sum(state[:, column] == EMPTY) == 0:
            raise FullColumnException(f"Column {column} is full, cannot insert piece.")

        # find last available space in chosen column and fill it
        row = np.sum(state[:, column] == EMPTY) - 1
        new_state = deepcopy(state)
        new_state[row, column] = self.tag2val[tag]
        return new_state

    def get_available_moves(self, state=None):
        if state is None:
            state = self.state

        return np.arange(7)[np.sum(state == EMPTY, axis=0) > 0]