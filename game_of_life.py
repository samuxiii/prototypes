import pygame
import numpy as np
import time

# Configurations
DARK = 25, 25, 25
WHITE = 255, 255, 255
size = width, height = 600, 600
nxC, nyC = 50, 50
dimCW = (width - 1) / nxC
dimCH = (height - 1) / nyC

# Cell states, Alive = 1; Dead = 0
gameState = np.zeros((nxC, nyC))

# Screen details
pygame.init()
screen = pygame.display.set_mode((size))
ev = pygame.event.get()

pygame.display.set_caption("Game of Life")


def drawSquare(x, y):
    global gameState
    # example point = ((0, 0), (10, 0), (10,10), (0, 10))
    point = ((x * dimCW, y * dimCH), \
             ((x + 1) * dimCW, y * dimCH), \
             ((x + 1) * dimCW, (y + 1) * dimCH), \
             (x * dimCW, (y + 1) * dimCH))

    # inverse gameState[x,y] == fill square or empy
    pygame.draw.polygon(screen, DARK, point, not gameState[x, y])


def updateSquare(newGameState, x, y):
    global gameState
    # Calculate neighbours
    neigh = gameState[(x - 1) % nxC, (y - 1) % nyC] + \
            gameState[x % nxC, (y - 1) % nyC] + \
            gameState[(x + 1) % nxC, (y - 1) % nyC] + \
            gameState[(x - 1) % nxC, y % nyC] + \
            gameState[(x + 1) % nxC, y % nyC] + \
            gameState[(x - 1) % nxC, (y + 1) % nyC] + \
            gameState[x % nxC, (y + 1) % nyC] + \
            gameState[(x + 1) % nxC, (y + 1) % nyC]

    # Any live cell with two or three live neighbours survives
    if gameState[x, y] == 1 and (neigh == 2 or neigh == 3):
        newGameState[x, y] = 1
    # Any dead cell with three live neighbours becomes a live cell
    elif gameState[x, y] == 0 and neigh == 3:
        newGameState[x, y] = 1
    # All other live cells die in the next generation. Similarly, all other dead cells stay dead
    else:
        newGameState[x, y] = 0


def updateGameState(pauseGame):
    global gameState
    newGameState = np.copy(gameState)

    for x in range(0, nxC):
        for y in range(0, nyC):
            drawSquare(x, y)
            if not pauseGame:
                updateSquare(newGameState, x, y)

    gameState = np.copy(newGameState)


### Cell initialization
gameState[1, 30] = 1
gameState[2, 30] = 1
gameState[3, 30] = 1
gameState[3, 29] = 1
gameState[2, 28] = 1

gameState[1, 40] = 1
gameState[2, 40] = 1
gameState[3, 40] = 1

#gameState = np.random.randint(0, 2, (50, 50))

def processClick():
    mouseClick = pygame.mouse.get_pressed()
    if sum(mouseClick) > 0:
        global gameState
        posX, posY = pygame.mouse.get_pos()
        celX, celY = int(np.floor(posX / dimCW)), int(np.floor(posY / dimCH))
        gameState[celX, celY] = not gameState[celX, celY]


### Main Loop ###
running = True
pauseGame = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            pauseGame = not pauseGame
        else:
            processClick()

    screen.fill(WHITE)
    updateGameState(pauseGame)
    pygame.display.flip()

    if not pauseGame:
        time.sleep(0.3)

pygame.quit()
