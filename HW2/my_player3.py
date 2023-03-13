from cmath import pi
from copy import deepcopy
from math import inf
import random

def readInput(path="input.txt"):
    with open(path, 'r') as f:
        lines = f.readlines()

        piece_type = int(lines[0])

        previous_board = [[int(x) for x in line.rstrip('\n')] for line in lines[1:5+1]]
        board = [[int(x) for x in line.rstrip('\n')] for line in lines[5+1: 2*5+1]]

        return piece_type, previous_board, board

def writeOutput(result, path="output.txt"):
    res = ""
    if result == "PASS":
        res = "PASS"
    else:
        res += str(result[0]) + ',' + str(result[1])

    with open(path, 'w') as f:
        f.write(res)

class Player():
    def __init__(self, piece_type, previous_board, board):
        self.piece_type = piece_type
        self.previous_board = previous_board
        self.board = board
    
    def detectNeighbors(self,i,j):
        neighbors = [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]
        return [point for point in neighbors if 0<=point[0]<5 and 0<=point[1]<5]

    def pieceScore(self, board, i, j, piece_type):
        score=0.0
        neighbors = self.detectNeighbors(i,j)
        for nei in neighbors:
            val = board[nei[0]][nei[1]]
            if val==0:
                score+=1
            if val==piece_type:
                score+=0.5
        return score
    
    def evaluateBoard(self, board, piece_type): #player piece is passed
        pscore = 0.0
        oscore = 0.0
        if piece_type==2:
            pscore+=2.5
        else:
            oscore+=2.5
        for i in range(5):
            for j in range(5):
                if board[i][j]==piece_type:
                    pscore+=1
                    if piece_type==2:
                        pscore+=self.pieceScore(board, i, j, piece_type)*0.25
                    else:
                        pscore+=self.pieceScore(board, i, j, piece_type)*1.25
                if board[i][j]==3-piece_type:
                    oscore+=1
                    oscore+=self.pieceScore(board,i,j,3-piece_type)*1.5
        return pscore-oscore
    
    def detectNeighborAlly(self,board,i,j):
        neighbor = self.detectNeighbors(i,j)
        return [piece for piece in neighbor if board[piece[0]][piece[1]]==board[i][j]]

    def allyDFS(self,board, i,j):
        stack = [(i, j)]  # stack for DFS serach
        ally_members = []  # record allies positions during the search
        while stack:
            piece = stack.pop()
            ally_members.append(piece)
            neighbor_allies = self.detectNeighborAlly(board, piece[0], piece[1])
            for ally in neighbor_allies:
                if ally not in stack and ally not in ally_members:
                    stack.append(ally)
        return ally_members
    
    def findLiberty(self,board,i,j):
        ally_members = self.allyDFS(board, i, j)
        for member in ally_members:
            neighbor = self.detectNeighbors(member[0],member[1])
            for piece in neighbor:
                if board[piece[0]][piece[1]] == 0:
                    return True
        return False
    
    def removePieces(self, board, pos):
        for piece in pos:
            board[piece[0]][piece[1]] = 0
        return board
    
    def findDeadPieces(self, board, piece_type):
        return [(i,j) for i in range(5) for j in range(5) if board[i][j]==piece_type and not self.findLiberty(board,i,j)]
    
    def removeDeadPieces(self, board, piece_type):
        dead_pieces = self.findDeadPieces(board,piece_type)
        if not dead_pieces: return board
        new_board = self.removePieces(board, dead_pieces)
        return new_board
    
    def ko(self, prev_board,next_board):
            for i in range(5):
                for j in range(5):
                    if prev_board[i][j] != next_board[i][j]:
                        return False
            return True
    
    def validMoveCheck(self, board, piece_type, i, j):
        if not 0<=i<5 and not 0<=j<5:
            return False
        if board[i][j] != 0:
            return False
        temp = deepcopy(board)
        temp[i][j] = piece_type
        if self.findLiberty(temp,i,j):
            return True
        temp = self.removeDeadPieces(temp, 3-piece_type)
        if not self.findLiberty(temp,i,j):
            return False
        prev_board = self.previous_board
        if self.ko(prev_board,temp):
             return False
        return True
    
    def findValidMoves(self, board, piece_type, decay):
        moves =  [(i,j) for i in range(5) for j in range(5) if self.validMoveCheck(board,piece_type,i,j) and not self.is_an_eye(board,i,j,piece_type)]
        try:
            moves = random.sample(moves,decay)
            return moves
        except:
            return moves
    
    def makeMove(self, board, move, piece_type):
        temp = deepcopy(board)
        temp[move[0]][move[1]] = piece_type
        temp = self.removeDeadPieces(temp, 3-piece_type)
        return temp

    def judge_winner(self, board):
        bs=0 #black - stones
        ws=2.5 #
        for i in range(5):
            for j in range(5):
                if board[i][j]==1: bs+=1
                if board[i][j]==2: ws+=1
        return 1 if bs > ws else 2

    def bestMove(self, board, piece_type,no_move):
        decay = 0
        if no_move==1:
            decay = 15
            valid_moves = self.findValidMoves(board, piece_type,decay)
        if no_move in range(2,10):
            decay=13
        if no_move in range(10,17):
            decay=14
        if no_move in range(17,24):
            decay=15
        valid_moves = self.findValidMoves(board, piece_type,decay)
        move_to_make = []
        best_score = -10000.0
        for move in valid_moves:
            next_state = self.makeMove(board, move, piece_type)
            if no_move==1:
                score = self.minimax(next_state,2,-10000,+10000,False,3-piece_type,no_move,decay)
            else:
                score = self.minimax(next_state,4,-10000,+10000,False,3-piece_type,no_move,decay)
            if score>best_score:
                best_score = score
                move_to_make = move
        return move_to_make

    def minimax(self, board, depth, alpha, beta, maxPlayer, piece_type, no_move,decay):
        piece = self.piece_type
        if no_move>=24:
            res = self.judge_winner(board)
            if maxPlayer and res==piece_type:
                return +10000.0
            if not maxPlayer and res!=piece_type:
                return +10000.0
            else:
                return -10000.0
        if depth == 0:
            return self.evaluateBoard(board,piece)
        if maxPlayer:
            maxEval = -inf
            valid_moves = self.findValidMoves(board,piece_type,decay)
            for move in valid_moves:
                next_state = self.makeMove(board, move, piece_type)
                eval = self.minimax(next_state, depth-1, alpha, beta, False, 3-piece_type, no_move+1,decay)
                if eval>maxEval:
                    maxEval = eval
                alpha = max(alpha,eval)
                if beta<=alpha:
                    break
            return maxEval
        else:
            minEval = inf
            valid_moves = self.findValidMoves(board,piece_type,decay)
            for move in valid_moves:
                next_state = self.makeMove(board, move, piece_type)
                eval = self.minimax(next_state, depth-1, alpha, beta, True, 3-piece_type, no_move+1,decay)
                if eval<minEval:
                    minEval = eval
                beta = min(beta,eval)
                if beta<=alpha:
                    break
            return minEval

    def is_an_eye(self, board,i,j,piece_type):
        neighbors = self.detectNeighbors(i,j)
        for neighbor in neighbors:
            if board[neighbor[0]][neighbor[1]] != piece_type:
                return False
        corners = [[i-1,j-1],[i-1,j+1],[i+1,j-1],[i+1,j+1]]
        friendly_corners = 0
        off_board_corners = 0
        for corner in corners:
            try:
                if board[corner[0]][corner[1]] == piece_type:
                    friendly_corners+=1
            except:
                off_board_corners+=1
        if off_board_corners>0:
            return off_board_corners+friendly_corners==4
        return friendly_corners>=3



if __name__=="__main__":
    piece_type, previous_board, board = readInput()
    player = Player(piece_type, previous_board, board)
    actions = []
    if board==[[0]*5]*5:
        N_MOVE = 0
        actions = (2,2)
    elif previous_board==[[0]*5]*5:
        N_MOVE = 1
        if board[2][2]==0:
            actions = (2,2)
        else:
            actions = player.bestMove(board,piece_type,N_MOVE)
    else:
        with open("move.txt","r") as f:
            N_MOVE = int(f.readline())
        actions = player.bestMove(board,piece_type,N_MOVE)
    if not actions:
        actions = "PASS"
    N_MOVE += 2
    with open("move.txt","w") as f:
        f.write(str(N_MOVE))
    with open("error.txt","a+") as f:
        f.write(str(actions))
    writeOutput(actions)