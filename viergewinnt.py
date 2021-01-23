import numpy as np
from copy import deepcopy

class VierGewinnt():

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
        pass


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



class Player():
    
    def __init__(self):
        self.tag = None
        self.tag_val = None
    
    def make_move(self, state):
        if self.tag_val is None:
            print('Please put the player in a game before making a move.')
            return state
        else:
            col = int(input('Choose a column (0-6): '))
            # find last available space in chosen column and fill it
            row = np.sum(state[:, col] == 0) - 1
            s = deepcopy(state)
            s[row, col] = self.tag_val
            
            return s
