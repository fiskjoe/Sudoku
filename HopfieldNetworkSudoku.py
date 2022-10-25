import numpy as np
import numpy.random as rnd
import math
import torch
import pandas as pd


def index_ijk(n, d):
  i = math.floor(n/d**2)
  j = math.floor((n-(i*d**2))/d) 
  k = n - (i*d**2) - (j*d)
  return i,j,k


def ijk_index(i, j, k, d):
  n = i*d**2 + j*d + k
  return n


def getMatAndVec(board, l_h=4, l_r=1, l_c=1, l_b=2, l_s=1):
  dimension = board.shape[0]

  hints = []
  for i in range(len(board)):
    for j in range(len(board[i])):
      if board[i][j] > 0:
        n = ijk_index(i, j, board[i][j]-1, dimension)
        hints.append(n)
        
  H = np.zeros((len(hints), dimension**3))
  for instance in range(len(H)):
    H[instance][hints[instance]] = 1

  R = np.zeros((dimension**2, dimension**3))
  for instance in range(len(R)):
    current_i = math.floor(instance/dimension)
    current_k = instance%dimension
    for n in range(len(R[i])):
      i,j,k = index_ijk(n, dimension)
      if (i==current_i and k==current_k):
        R[instance][n] = 1
  
  C = np.zeros((dimension**2, dimension**3))
  for instance in range(len(C)):
    current_j = math.floor(instance/dimension)
    current_k = instance%dimension
    for n in range(len(C[i])):
      i,j,k = index_ijk(n, dimension)
      if (j==current_j and k==current_k):
        C[instance][n] = 1

  B = np.zeros((dimension**2, dimension**3))
  for instance in range(len(B)):
    current_i = math.floor(instance/dimension)
    current_j = instance%dimension
    for n in range(len(B[i])):
      i,j,k = index_ijk(n, dimension)
      if (i==current_i and j==current_j):
        B[instance][n] = 1  

  S = np.zeros((dimension**2, dimension**3))
  for instance in range(len(S)):
    region = math.floor(instance/dimension)
    current_k = instance%dimension
    for n in range(len(S[i])):
      i,j,k = index_ijk(n, dimension)
      same_region = (region%3==math.floor(j/3) and (math.floor(region/3)==math.floor(i/3)))
      if (same_region and k == current_k):
        S[instance][n] = 1       
  
  v1_n2 = np.ones(dimension**2)
  v1_n3 = np.ones(dimension**3)
  v1_h = np.ones(len(hints))
  
  Q = (l_r * R.T @ R) + (l_c * C.T @ C) + (l_b * B.T @ B) + (l_s * S.T @ S) + (l_h * H.T @ H)
  q = 2 * ( (l_r * v1_n2 @ R) + (l_c * v1_n2 @ C) + (l_b * v1_n2 @ B) + (l_s * v1_n2 @ S) + (l_h * v1_h @ H) )

  W = -0.5 * Q ; 
  np.fill_diagonal(W, 0)
  t = 0.5 *(Q @ v1_n3 - q)

  return W, t
  

def simulatedAnnealing (s, W, t, Th=10, Tl=0.5, numT=21, passes=1500):
  n = len(vecS)
  for T in np.linspace(Th, Tl, numT):
    for r in range(passes):
      for i in range(n):
        q = 1 / (1 + np . exp ( -2/ T * ( W[i] @s - t[i])))
        z = rnd.binomial(n=1, p=q )
        s[i] = 2*z - 1
  energy = (-0.5 * s.T @ W @ s) + (t.T @ s)
  return s, energy
  


# -=- Testing solution on example Sudoku problems -=-

def printBoard(boardList):
  dimension = 9
  output = np.zeros((dimension, dimension), dtype=np.int8) 
  squareAcc = []
  oldNj = 0
  for n in range(len(boardList)):
    ni = math.floor(n/dimension**2)
    nj = math.floor((n-(ni*dimension**2))/dimension)
    nk = n - (ni*dimension**2) - (nj*dimension)

    if (nj != oldNj) or (n==len(boardList)-1):
      if (n==len(boardList)-1): 
        squareAcc.append(boardList[n])
      if sum(squareAcc) == -dimension:
        output[oldNi][oldNj] = 0
      else:
        output[oldNi][oldNj] = squareAcc.index(1) + 1
      squareAcc = []
    
    squareAcc.append(boardList[n])

    oldNj = nj   
    oldNi = ni

  return output

def guessesCorrect(guess, answer):
  guess = printBoard(guess);
  n = len(guess)
  total = 0
  for i in range(n):
    for j in range(n):
      if (guess[i][j] == int(answer[i*n+j])):
        total += 1
  return(total)
  
def stringToBoard(board):
  n = int(np.sqrt(len(board)))
  b = np.zeros((n, n), dtype=np.int8)
  for i in range(n):
    for j in range(n):
      b[i][j] = int(board[i*n+j])
  return b
  
  
easy_q = [
          "061030020050008107000007034009006078003209500570300900190700000802400060040010250",
          "100830002570001000000509064704008590003010400051400306360704000000600079800052003",
          "030007004602041000050030967040003006087000350900700020718020040000160809400500030",
          "085000210094012003000300704503409000040206030000103907608005000100840360027000890"
]

easy_a = [
          "761934825354628197928157634219546378483279516576381942195762483832495761647813259",
          "149836752576241938238579164724368591683915427951427386362794815415683279897152643",
          "839657214672941583154832967541283796287496351963715428718329645325164879496578132",
          "385764219794512983216398754573489126941276538862153947638925471159847362427631895"
]

q_num = 3
question = easy_q[q_num]
answer = easy_a[q_num]

board = stringToBoard(question)

n3 = board . shape [0]**3

matW, vecT = getMatAndVec(board, l_h=4, l_b=2, l_r=1, l_c=1, l_s=1)

vecS = rnd . binomial ( n =1 , p =0.05 , size = n3 ) * 2 - 1
vecS_result, energy = simulatedAnnealing(vecS, matW, vecT, Th=10, Tl=0.5, numT=21, passes=500)   

print("Original Problem:")
print(stringToBoard(question))
print("")

print("Neural Network Solution:")
print(printBoard(vecS_result))
print("")

print("Actual Answer:")
print(stringToBoard(answer))
