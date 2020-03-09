def triedrdf():
    """
    Le module triedrdf regroupe des méthodes pour la reconnaissance de forme :
        bayesrule  : Règle de décision de Bayes
        kppv       : Algorithme des k-plus proches voisins
        kmoys      : Algorithme des k-moyennes (kmeans)
        kclassif   : Classification de données par proximité à des référents
    """
    return None

import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from   matplotlib import cm
import triedpy.triedtools as tls

#========================= Règle de Bayes =========================
def bayesrule(Prob,loss=None) :
    ''' BAYESCLASSES = bayesrule(Prob,loss) is the vector of classification.
    | p is the (nxc) matrix of the posterior probabilities.
    | loss is an optional (cxc) matrix of classication costs where loss(i,j)
    | is the cost of classifying in class j a pattern of class i. 
    | If loss is omitted, (0,1) loss are used by default.
    '''
    nX, c = np.shape(Prob)
    if loss is None :
        loss  = np.ones((c,c)) - np.eye(c,c);
    # classification
    Risk  = np.dot(Prob,loss);
    BAYESCLASSES = np.argmin(Risk.T,0)+1;
    return BAYESCLASSES

#==================== k plus proches voisins ======================
def kppv (X,Xi,labelXi,k,dist=0,votopt=0) :
    '''XCLASSE = kppv (X,Xi,labelXi,k,dist,votopt)
    | Algorithme des k plus proches voisins (kppv)
    | En entrée :
    | X       : Un ensemble de données (matrice nX x d) dont on veut classer les
    |           éléments (lignes) par l'algorithme kppv (ensemble de test)
    | Xi      : Ensemble de référence (d'apprentissage); matrice (nXi x d)
    | labelXi : les indices de classe de référence (i.e. des éléments de l'ensemble
    |           de référence). Ces indices doivent commencer à 1 (>0) et non pas
    |           à partir de zéro.; vecteur colonne (nXi x 1)
    | k       : Le nombre de plus proches voisins à considérer
    | dist    : Distance à utiliser: 0 : Euclidienne (par défaut); sinon: Mahalanobis
    | votopt  : Définit l'option en cas d'egalité du vote majoritaire :
    |           0 : En cas d'égalité de classe c'est alors la 1ère qui est retenue
    |           (c'est l'option par défaut), sinon, un tirage aléatoire est effectué.
    | En sortie :
    | XCLASSE   : Classes des éléments de X; c'est un vecteur colonne (nX x 1)
    '''
    if min(labelXi)<=0 :
        print("kppv: Les indices de classe de référence (labelXi) doivent être > 0");
        sys.exit(0);
        
    nX,  d1 = np.shape(X);
    nXi, d2 = np.shape(Xi);
    c       = max(labelXi);

    if d1 != d2 :
        print("kppv: X doit avoir la même dimension que Xi");
        sys.exit(0);
    if np.size(Xi,0) != np.size(labelXi) :
        print("kppv, Xi et labelXi doivent avoir le même nombre d'éléments (i.e. de lignes)");
        sys.exit(0);
      
    labelXi = labelXi-1;    # parce que les indices commence à 0 ...?
    XCLASSE = np.zeros(nX); # Clasification des éléments par kppv

    if dist!=0 : # Distance de MAHALANOBIS
        SIGMA = np.zeros((c,d1,d2)); # Init.  des matrices de COVARIANCE (par classe)
        for i in np.arange(c) :      # Calcul des matrices de COVARIANCE (par classe)
            ICi         = np.where(labelXi==i)[0]; # Indices des élts de la classe i
            Xiclasse    = Xi[ICi,:];          # Ens de Ref pour la classe i
            sigma       = np.cov(Xiclasse.T, ddof=1);
            sigmamoins  = np.linalg.inv(sigma);
            SIGMA[i,:,:]= sigmamoins;

    # DECISION
    for i in np.arange(nX) : # Pour chaque élément de l'ensemble de TEST
        D = np.zeros(nXi);

        if dist==0 : # Distance de Euclidienne (on ommet la metrique I)
            for j in np.arange(nXi) : # Pour chaque elt de l'ens de référence
                D[j] = np.dot(X[i,:]-Xi[j,:] , X[i,:]-Xi[j,:]);
        else : # Distance de Mahalanobis
            for j in np.arange(nXi) : # Pour chaque elt de l'ens de référence
                cl   = labelXi[j];
                M    = np.dot(X[i,:]-Xi[j,:], SIGMA[cl,:,:]);
                D[j] = np.dot(M, X[i,:]-Xi[j,:]);            

        # Vote majoritaire
        # Tri des distances dans l'ordre du +petit au +grand
        I = sorted(range(len(D)), key=lambda k: D[k])
        C         = labelXi[I];   # On ordonne les classes selon ce tri
        classeppv = C[0:k];    # On garde les k premières classes qui correspondent 
                               # donc au kppv dans l'ens de référence                      
        # Vote majoritaire :
        XCLASSE[i] = tls.avotemaj(classeppv,votopt=votopt)

    XCLASSE = XCLASSE+1; # Pour revenir à l'indicage initial.
    return XCLASSE

#=========================== k-moyennes ===========================
def kmoys (X,k,spause=0,pvisu=0,cmap=cm.jet,markersize=8,fontsize=11) :
    '''PROTO, CLASSE = kmoys (X,k,spause,pvisu)
    | Algorithme des k-moyennes (kmeans)
    | En entrée :
    | X       : Est l;'ensemble de donnees. dim(X)=(N,p)
    | k       : Est le nombre de classes (<=N).
    | [spause]: (optionel) Nombre de secondes (meme fractionnaire) de pause  
    |           pour ralentir le code afin d'avoir le temps de voir l'évolution  
    |           des protos sur la figure. Passer 0 si on préfère un temps 
    |           d'exécution non ralenti (c'est la valeur par defaut).
    | [pvisu] : (optionel) Vecteur de dimension 2 indiquant le plan de 
    |           visualisation. pvisu[0] est l'abscisse, et pvisu[1] l'ordonnée.
    |           Par défaut le plan 1-2 est utilisé.
    | cmap    : La map de couleur
    | markersize : La taille des marqueurs
    | fontsize   : La taille du text
    | En sortie :
    | PROTO  : Est la matrice de coordonnees des prototype. dim(proto)=(k,p)
    | CLASSE : Une vecteur colonne qui contient le numéro de proto définissant
    |          ainsi une classe pour chaque individu.
    '''
    N = np.size(X,0);
    
    # plan de visualisation 
    if pvisu == 0 :
        a = 0; o = 1;   # par defaut
    else :
        a=pvisu[0]-1; o=pvisu[1]-1; 

    # Initialisation du vecteur des classes des individus
    CLASSE = np.zeros(N);

    # Map de couleur pour le plot des protos et de leurs trajectoires
    #cmap = plt.cm.jet
    Tcol  = cmap(np.arange(1,256,round(256/k)))       # k lignes de couleur
    
    # Tirage des k prototypes au hasard
    Iprot = np.random.permutation(N); 
    PROTO = X[Iprot[range(k)],:]; #print("\nproto=",proto);
    prevprot = PROTO;
    
    plt.figure();
    plt.ion()   # interactive graphics on
    plt.plot(X[:,a], X[:,o],'+k');
    
    # Loop initialisation ---------------
    oldcritere = -1;
    critere    =  0;
    print("   Critère    Critère normalisé par");
    print("               le nombre de données:");
    #
    # Tant que la convergence n'est pas atteinte
    # On affecte chaque point a la classe la plus proche
    while oldcritere != critere :
        oldcritere = critere;
                
        # Calcul d'une matrice des inerties intra
        distance = np.zeros((N,k))
        for i in range(N) :
            for j in range(k) :
                C = X[i,:] - PROTO[j,:];
                distance[i,j] = np.dot(C,C);
                
        # Calcul du critere et de la classe d'appartenance
        critere = 0;
        for i in range(N) :
            di = distance[i,:];
            minligne = min(di)
            CLASSE[i] = np.argmin(di) # !!! à partir de 0, on fera +1 si necessaire à la fin ?
            critere = critere + minligne;

        # Positions des nouveaux prototypes
        for i in range(k) :
            Ic = np.where(CLASSE==i);
            if np.size(Ic) > 0 :
                PROTO[i,:] = np.mean(X[Ic,:],1);

        # Affichage
        print("% .10f   % .10f" % (critere,critere/N));
        for i in range(k):
            plt.plot([prevprot[i,a], PROTO[i,a]], [prevprot[i,o], PROTO[i,o]],\
                     "o-",linewidth=3,color=Tcol[i,:],markersize=markersize);
            #plt.plot(PROTO[i,a], PROTO[i,o],"o-");
        prevprot = PROTO.copy();
        time.sleep(spause);
        plt.draw();
    #
    # fin du while
    #--------------------------------------------------------------
    for i in range(k) :
        Ic = np.where(CLASSE==i);
        plt.plot(X[Ic,a],X[Ic,o],"*",color=Tcol[i,:],markersize=markersize);
        plt.plot(PROTO[i,a], PROTO[i,o],"s",color=[0,0,0]);
        plt.text(PROTO[i,a], PROTO[i,o],str(i+1), fontsize=fontsize);
    plt.axis("tight");
    plt.xlabel("x%d" %(a+1));
    plt.ylabel("x%d" %(o+1));
    plt.title("Kmeans algorithme on data with k=%d" % (k));
    #plt.ioff();
    #
    #--------------------------------------------------------------
    CLASSE = CLASSE + 1 # On retourne des N° de classe numérotée à partir de 1
    return PROTO, CLASSE

#============================ kclassif ============================
def kclassif (X,proto,clasprot=None,opt=0) :
    '''KLASS = kclassif (X,proto,clasprot,opt)
    | Associe aux éléments de X, la classe du prototype (référent) le plus proche
    | (au sens euclidien).
    | En entrée :
    | X     : L'ensemble des données à classer
    | proto : Les référents auquels les données doivent être associées selon leur
    |         proximité
    | clasprot : Classe des référents. Les données associées à un référent, par
    |         proximité, hériterons de (se veront attribuer), la classe du référent.
    |         Si ce paramètre n'est pas renseigné, on attribue d'office une classe
    |         au prototype, dans l'ordre, et à partir de 1. 
    | opt   : Permet ou pas la prise en compte des prototypes associés à une classe
    |         nulle (=0). En effet, un référent qui n'aurait capté aucune donnée
    |         peut avoir sa valeur de classe dans clasprot = 0.
    |         si opt = 0 : on affectera, à une donnée, la classe du proto qui lui
    |         est le plus proche même si cette classe est 0 (c'est le cas par defaut),
    |         sinon, on ira chercher le proto le plus proche dons la classe est
    |         différente de 0.
    | En sortie :
    | KLASS : Classes affectées aux éléments de X (correspondant pour chacun à celle
    |         du protos qui lui est le plus proche)        
    '''
    k, q = np.shape(proto);
    if clasprot is None :
        clasprot = (np.arange(k)+1).astype(int);
    else :
        if np.size(clasprot) != k :
            print("kclassif: clasprot must be same length as proto (each proto must have a class in clasprot)");
            sys.exit(0);        
    N, p     = np.shape(X);
    if p != q :
        print("kclassif: data X and proto must have the same dim");
        sys.exit(0);
 
    KLASS    = np.zeros(N);  # Init classes
    distance = np.zeros(k);  # Init distances
    
    for i in np.arange(N) :
        for j in np.arange(k) :
            C = X[i,:] - proto[j,:];
            distance[j] = np.dot(C,C);

        iprot = np.argmin(distance);  # indice du proto le plus proche de i

        if clasprot[iprot]==0 and opt!=0 :
            # On va rechercher le proto le plus proche dont la classe est > 0;
            b = np.argsort(distance);   # b: les indices du + petit au + grd
            ii=0;
            while clasprot[b[ii]]<=0 and ii<k :
                ii=ii+1;
            if ii<k : 
                KLASS[i] = clasprot[b[ii]];
            #sinon KLASS[i] est déjà à 0 par son initialisation. 
        else :
            KLASS[i] = clasprot[iprot];
    return KLASS

#==================================================================
#
