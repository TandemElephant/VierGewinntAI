import numpy as np
import random
from scipy.signal import convolve2d
from copy import deepcopy

EMPTY = 0


class FullBoardException(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class VierGewinnt:
    def __init__(self, player1, player2):

        self.player1 = player1
        self.player1.tag = 'X'
        self.player1.tag_val = 1
        self.player2 = player2
        self.player2.tag = 'O'
        self.player2.tag_val = -1

        # init game
        self.state = np.zeros((6, 7))
        self.winner = None
        self.turn = 'X'
        self.turn_player = self.player1

    def init_game(self):
        self.state = np.zeros((6, 7))
        self.winner = None
        self.turn = 'X'
        self.turn_player = self.player1
        
    def play_game(self):
        while self.winner is None:
            
            if type(self.turn_player) == Player:
                self.print_game()
                print(f"{self.turn_player.tag}'s turn.")

            self.state = self.play_move()

            self.check_winner()
            if self.winner is not None:
                break

        self.print_game()
        print(f"The winner is: {self.winner}")

    def play_games_for_learning(self, n_games):

        for i in range(n_games):

            while self.winner is None:
                self.state = self.play_move(learn=True)
                self.check_winner()

                if self.winner is not None:
                    break

                self.state = self.play_move(learn=True)
                self.check_winner()

            # update loser's state
            self.state = self.play_move(learn=True)
            self.state = self.play_move(learn=True)
            # update winner's state
            self.state = self.play_move(learn=True)
            self.state = self.play_move(learn=True)

    def check_winner(self):
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

        for win_pos in winner_positions:
            convolved_state = convolve2d(self.state, win_pos, mode='valid')

            if (convolved_state == 4).any():
                self.winner = self.player1.tag
                break
            elif (convolved_state == -4).any():
                self.winner = self.player2.tag
                break
            elif (convolved_state != EMPTY).all():
                self.winner = "Nobody (tie)"
                break

            return self.winner

    def play_move(self, learn=False):

        if learn is True:
            new_state = self.turn_player.make_move_and_learn(self.state, self.winner)
        else:
            new_state = self.turn_player.make_move(self.state)

        self.next_player()
        return new_state

    def next_player(self):
        if self.turn == 'X':
            self.turn = 'O'
            self.turn_player = self.player2
        else:
            self.turn = 'X'
            self.turn_player = self.player1

    def print_game(self):
        print(self.state)


class Player:
    def __init__(self):
        self.tag = None
        self.tag_val = None
    
    def make_move(self, state):
        if self.tag_val is None:
            print('Please put the player in a game before making a move.')
            return state
        else:
            col = int(input('Choose a column (0-6): '))
            new_state = self.insert_piece(state, col)
            return new_state

    def insert_piece(self, state, column):
        # find last available space in chosen column and fill it
        row = np.sum(state[:, column] == EMPTY) - 1
        new_state = deepcopy(state)
        new_state[row, column] = self.tag_val
        return new_state


class DeepAgent(Player):
    def __init__(self, exploration_factor=0.2):
        super().__init__()
        self.exp_factor = exploration_factor

    def make_move(self, state):

        available_columns = np.arange(7)[np.sum(state == EMPTY, axis=0) > 0]

        # no available move
        if len(available_columns) == 0:
            raise FullBoardException("Cannot make move, board is full.")

        # random exploration move
        if random.random() < self.exp_factor:
            col = random.choice(available_columns)
            new_state = self.insert_piece(state, col)

        # make optimal move
        else:
            optimal_move_value = -np.inf
            optimal_moves = []
            for col in available_columns:
                move_value = self.calc_move_value(state, col)
                if move_value > optimal_move_value:
                    optimal_moves = [col]
                    optimal_move_value = move_value
                elif move_value == optimal_move_value:
                    optimal_moves.append(col)

            optimal_move = random.choice(optimal_moves)
            new_state = self.insert_piece(state, optimal_move)

        return new_state

    def make_move_and_learn(self, state, winner):

        self.learn_state(state, winner)

        if winner is None:
            new_state = self.make_move(state)
        else:
            new_state = state

        return new_state

    def calc_state_value(self, state):
        pass

    def calc_move_value(self, state, column, iteration=1):
        pass