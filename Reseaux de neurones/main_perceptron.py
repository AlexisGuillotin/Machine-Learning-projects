"""
Nom : Alexis Guillotin, Clément Marie--Brisson
UE : USSI5E - Machine Learning
Projet : Réseaux de neuronnes
Date de rendu : 

"""
import numpy as np
import matplotlib.pyplot as plt


# ► 1.1  Mise en place d’un perceptron simple
def perceptron_simple(x,w,active):
    # x → inputs
    # w → wheight
    # active → if activation function (phi) is used or not

    #variable initialisation
    #x.insert(0,1) # with x[0] = 1
    x = np.insert(x,0,1)
    y=0

    for i in range( len(w) ):
        y+=w[i]*x[i]

    if active==0:
        return np.sign(y)
    elif active==1:
        return np.tanh(y)
    else:
        return 'active n\'est pas correctement renseigné {0;1}'

# ► 1.2 Etude de l'apprentissage
    # ► 1.2.1 Programmation apprentissage Widrow-hoff

# Links :
#   • https://www.cs.princeton.edu/courses/archive/spring13/cos511/scribe_notes/0409.pdf
#   • https://www.cs.princeton.edu/courses/archive/spring13/cos511/scribe_notes/0411.pdf

# Variables explanations :
#   • x:   contient  l’ensemble  d’apprentissage.  C’est  une  matrice  `a  2  lignes  et  n  colonnes.
#   • yd: yd[i]  indique  la  r´eponse  d´esir´ee  pour  chaque  ´el´ement  x[:,i].  yd  est  un  vecteur  de  1  ligne  et  n colonnes  de  valeurs  +1  ou  -1  (classiﬁcation  à  2  classes).
#   • Epoch: the number of times the model is exposed to the training set
#   • Batch_size: the number of training instances observed before the optimizer performs a weight update

def apprentissage_widrow(x,yd,epoch,batch_size):
    #err=pow(10,9999)
    erreur = np.zeros((epoch,1))
    n = 1 #learning rate 
    q=0
    while q<epoch:
        #if q>0:
        #    print( "epoch : ", q, ", erreur = ", erreur[q])
        for i in range(len(yd)):
            w = np.random.rand( 1+len(x)) #initialize w1=0
            # → Predict y[t]=w[t]*x[t]
            y=perceptron_simple(x[:,i],w,0)

            # → Observe y[t] : continue until we reach batch_size, or "convergence".
            for j in range(batch_size):
                # → incur loss of (ydt − y)²
                #for k in range(len(yd)):
                    #print(err,"+=", pow( yd[i]-y ,2) )
                    #err+=pow( yd[k]-y ,2)/len(yd)
                erreur[q]+=pow( yd[j]-y ,2)

                # → updating w: wt+1 = wt − n*(wt * xt − yt)xt;
                for k in range(len(w)-1):
                    for l in range(len(x)):
                        w[k+1] = w[k] - n*erreur[k]*x[l,k]

        #print("\t w = ", w, ", erreur = ",erreur[q])
        q+=1
    return w,erreur

# ► 1.3.1 Mise en place d'un perceptron multicouche
def multiperceptron(x,w1,w2):
    """
    Computes the output of a multi-layer perceptron with 1 neuron on the output layer and 2 neurons on the hidden layer for an
    on the hidden layer for an input element of R2. The activation function is phi(x) = 1/(1+exp(-x )
    :param x: input of the neural network. It is a 2 lines vector.
    :param w1: synaptic weights of the 2 neurons of the hidden layer. It is a matrix with 3 rows and 2 columns.
    The first column w1(:,1) corresponds to the 1st neuron of the hidden layer and w1(:,2) corresponds to the 2nd neuron of the hidden layer.
    :param w2: synaptic weights of the output layer neuron. It is a 3-line vector.
    :return: y is a scalar corresponding to the output of the neuron.
    """
    x = np.insert(x,0,1)

    # Compute the output of the hidden layer
    z = np.zeros((3,1))
    for i in range(2):
        z[i] = 1/(1+np.exp(-np.dot(w1[:,i].T,x)))
    # Compute the output of the output layer
    y = 1/(1+np.exp(-np.dot(w2.T,z)))
    return y

# ► 1.3.2 *Programmation apprentissage multicouches*
"""
La fonction multiperceptron widrow(x,yd,Epoch,Batch size) renvoie les poids w1 et w2 obtenus en apprenant selon la règle d'apprentissage par descente de gradient.
Nous étudions le cas d'un perceptron multicouche avec 1 neurone sur la couche de sortie et 2 neurones sur la couche cachée pour un élément d'entrée de R2. 
La fonction d'activation est phi(x) = 1/(1+exp(-x)) .

Les paramètres :
    - x contient l'ensemble d'apprentissage. C'est une matrice à 2 lignes et n colonnes.
    - yd(i) indique la réponse souhaitée pour chaque élément x( :,i). yd est un vecteur de 1 ligne et de n
    colonnes avec les valeurs +1 ou 0 (classification à 2 classes).
    - Epoch : le nombre d'itérations sur l'ensemble d'apprentissage.
    - Batch_size : le nombre d'individus de l'ensemble d'apprentissage traités avant la mise à jour des poids.
Le résultat :
    - w1 : contient les poids synaptiques des 2 neurones de la couche cachée. Il s'agit d'une matrice de 3
    lignes et 2 colonnes. La première colonne w1( :,1) correspond au 1er neurone de la couche cachée et w1( :,2)
    correspond au 2ème neurone.
    - w2 : contient les poids synaptiques du neurone de la couche de sortie. Il s'agit d'un vecteur à 3 lignes.
    - error : contient l'erreur cumulée calculée pour l'ensemble complet de l'apprentissage.
    - Tester la fonction multi-perceptron avec le cas XOR. 
"""
def phi(x):
    return 1/(1+np.exp(-x))

def multiperceptron_widrow(x,yd,Epoch,Batch_size):
    n=x.shape[1]
    x = np.insert(x, 0, [1,1,1,1], axis=0)

    w1=np.random.rand(3,2)
    w2=np.random.rand(3,1)
    error=np.zeros(Epoch)

    for i in range(Epoch):
        for j in range(0,n,Batch_size):
            x_batch=x[:,j:j+Batch_size]
            yd_batch=yd[:,j:j+Batch_size]
            
            y_batch=phi(np.dot(w1.T,x_batch))
            y_batch=np.vstack((y_batch,np.ones((1,Batch_size))))
            
            y=phi(np.dot(w2.T,y_batch))
            
            e=yd_batch-y
            error[i]=error[i]+np.sum( e**2)
            
            delta_w2=np.dot(y_batch,e.T)
            delta_w1=np.dot(x_batch,np.dot(w2[0:2,:],e).T)
            w2=w2+delta_w2
            w1=w1+delta_w1
    return w1,w2,error

def test_multiperceptron_widrow():
    x=np.array([[0,0,1,1],[0,1,0,1]])
    yd=np.array([[0,1,1,0]])
    w1,w2,error=multiperceptron_widrow(x,yd,100,4)
    print('\tw1:\n',w1)
    print('\tw2:\n\t',w2.T)
    #Tracé de l'évolution de l'erreur au fil du temps
    plt.plot(error)
    plt.xlabel('Epochs')
    plt.ylabel('Erreur mesurée')
    plt.title('Evolution de l\'erreur au fil des itérations')
    plt.show()
    #Tracé de la sortie du multiperceptron de widrow:
    plt.plot(w1,w2)
    plt.scatter(0,0)
    plt.scatter(1,0)
    plt.scatter(1,1)
    plt.scatter(0,1)
    # Partie à vérifier: mettre un titre, titre pour les axes, est-ce bien ce qu'il faut tracer?
    plt.show()

def main():
    # ► 1.1  Mise en place d’un perceptron simple
    print("► 1.1  Mise en place d’un perceptron simple")
    x = [0,0]
    w = [0, 1, 1]
    active = 0
    y = perceptron_simple(x,w,active)
    print( "Résultat perceptron simple : ", y )
        #Ploting of the line:
    a=-1
    b=0.5
    t=[-0.5,1.5]
    z=[a*t[0]+b,a*t[1]+b]
    plt.plot(t,z)
    plt.scatter(0,0)
    plt.scatter(1,0)
    plt.scatter(1,1)
    plt.scatter(0,1)
    plt.grid()
    plt.show()
#────────────────────────────────────────────────────────
    # ► 1.2 Etude de l'apprentissage
    print("► 1.2 Etude de l'apprentissage:")
    # ► 1.2.2 Test 1 simple :
    print("► 1.2.2 Test 1 simple:")        
        # Loading 'p2_d1.txt' data: initialising x.  
    x = np.loadtxt('p2_d1.txt')
        # initialising yd as 25 firsts elements equals to 1 and 25 lasts equals to -1.
    yd = np.ones([50,1], dtype=int)
    yd[25:,:] *= -1
    epoch = 5
    batch_size = 3 
    w_test1,erreur_test1 = apprentissage_widrow(x,yd,epoch,batch_size)
    print("w test 1: ",w_test1,"\nerreur test 1:", erreur_test1.transpose())
    # ► 1.2.3 Test 2
    print("► 1.2.3 Test 2")  
    x = np.loadtxt('p2_d2.txt')
    w_test2,erreur_test2 = apprentissage_widrow(x,yd,epoch,batch_size)
    print("w test 2: ",w_test2,"\nerreur test 2:", erreur_test2.transpose())
    
    print("CONCLUSION: ")
    if erreur_test1.max()>erreur_test2.max():
        print('\t• L\'erreur maximale est dans le test 1: ', erreur_test1.max())
    else:
        print('\t• L\'erreur maximale est dans le test 2: ', erreur_test2.max())
    if erreur_test1.min()<erreur_test2.min():
        print('\t• L\'erreur minimale est dans le test 1: ', erreur_test1.min())
    else:
        print('\t• L\'erreur minimale est dans le test 2: ', erreur_test2.min())
    if erreur_test1.mean()<erreur_test2.mean():
        print('\t• L\'erreur moyenne la plus basse est dans le test 1: ', erreur_test1.mean())
    else:
        print('\t• L\'erreur moyenne la plus basse est dans le test 2: ', erreur_test2.mean())
#────────────────────────────────────────────────────────
    # ► 1.3 Perceptron multicouches
    print('► 1.3 Perceptron multicouches')
    # ► 1.3.1 Mise en place d'un perceptron multicouche
    print('► 1.3.1 Mise en place d\'un perceptron multicouche')
    x = w1 = w2 = None
    x = np.array([[1],[2]])
    w1 = np.array([[2,0.5],[1,-1],[-1,1]])
    w2 = np.array([[0.5],[-0.5],[2]])
    y = multiperceptron(x,w1,w2)
    print('\ty: ', y)
    #faire la comparaison avec le test papier !!
    print("faire le test papier")
    
    # ► 1.3.2 *Programmation apprentissage multicouches*
    print('► 1.3.2 *Programmation apprentissage multicouches*')
    w1=w2=erreur = None    
    test_multiperceptron_widrow()

#────────────────────────────────────────────────────────


if __name__ == '__main__':
    main()