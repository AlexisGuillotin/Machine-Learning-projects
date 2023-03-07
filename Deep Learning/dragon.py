import numpy as np
from termcolor import colored

class dragon():
    case_depart = (0,0)
    position = case_depart
    def Plateau():
        plateau = [[1,0,0,0],[2,0,2,0],[0,0,0,2],[0,2,0,3]]
        return plateau
    
    def Print_plateau(plateau, position):
        x, y = position
        for i in range(0,4):
            for j in range(0,4):
                if x==i and y==j:
                    print(colored(plateau[i][j],'red'), end='')
                else:
                    print(plateau[i][j], end='')
            print('')
    

    ''''
    Les parametres :
    - La variable action contient l'action eectuee (0,3).
    - La variable position contient la position de l'agent.
    - La variable space contient l'organisation du plateau.
    Le resultat :
    - La variable position est la nouvelle position.
    - La variable Reward est la recompense obtenue (un reel).
    - La variable fin est un boolean indiquant si la partie est finie.
    '''
    def application_action(action,position,space):
        x, y = position
        reward = 0
        fin = False
        case = 0
        if action == "droite":
            if y+1 <=3:
                case = space[x][y+1]
                position = (x,y+1)
                iteration+=1
            else:
                print("Mouvement impossible")
                return position, reward, fin

        if action == "gauche":
            if y-1 >=0:
                case = space[x][y-1]
                position = (x,y-1)
                iteration+=1
            else:
                print("Mouvement impossible")
                return position, reward, fin

        if action == "haut":
            if x-1 >=0:
                case = space[x-1][y]
                position = (x-1,y)
                iteration+=1
            else:
                print("Mouvement impossible")
                return position, reward, fin

        if action == "bas":
            if x+1 <=3:
                case = space[x+1][y]
                position = (x+1,y)
                iteration+=1
            else:
                print("Mouvement impossible")
                return position, reward, fin
        #Case dragon
        if case == 2:
            print("Case "+str(position)+ "est une case dragon. Retour au début.")
            position = (0,0)
            if x==2 and y ==3:
                reward -= 1
        
        #Case Jewel
        if case == 3:
            reward = 1
            iteration = 0
            position = (0,0)
            fin == True
            print("Gagné !")
        dragon.Print_plateau(space,position)
        print('-----------')
        return position, reward, fin
    
    def choose_action(state,epsilon,mat_q):
        if epsilon<mat_q:
            return 1
        else:
            return 0
        
    def onestep(mat_q,state,epsilon):
        pass

res = dragon.Plateau()


position = (0,0)

mat_q = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
state = position
NBaction = 100
iteration = 0
epsilon = NBaction/(NBaction+iteration)


action = dragon.choose_action(state,epsilon,mat_q)
new_q,new_state = dragon.onestep(mat_q,state,epsilon)


#Tour 1
position, reward, fin = dragon.application_action("droite", position,res)

#Tour 2
position, reward, fin = dragon.application_action("bas", position,res)

#Tour 3
position, reward, fin = dragon.application_action("bas", position,res)

#Tour 4
position, reward, fin = dragon.application_action("droite", position,res)

#Tour 5
position, reward, fin = dragon.application_action("droite", position,res)

#Tour 6
position, reward, fin = dragon.application_action("droite", position,res)


