def triedtools () :
    """
    Le module triedtools regroupe les différentes méthodes
    suivantes :
        centree      : Centrage de données
        centred      : Centrage et réduction des données
        decentred    : Décentrage et déreduction de données
        normrange    : Normalisation sur un intervalle [a, b]
        linreg       : Régression linéaire
        errors       : Calcul d'erreurs     
        matconf      : Matrice de confusion
        plotby2      : Affichage de données par variables deux à deux.
        avotemaj     : Vote majoritaire      
        kvotemaj     : Vote majoritaire multiple
        classperf    : Performance de classification.
        shownpat2D   : Affichage d'un vecteur sous forme d'une image 2D
        rota2D       : Matrice de rotation 2D
        gen2dstd     : Génération de données simulées 2D selon une loi N(m,s)
        gen2duni     : Génération d'un ensemble de données uniformes 2D
        intarray2str : Transformation d'un tableau d'entiers en tableau de strings
        vdist        : Calcul de distance
    Matlab like function (+|-) :
        unique       : Retourne les occurrences, sans répétition, d'un tableau 2D
        ismember     : Indique si un vecteur (ligne) est contenu dans un tableau 2D
        unifpdf      : Fonction de densité de la loi uniforme
    """
    return None

import sys
import numpy as np
import numpy.matlib as nm
import matplotlib.pyplot as plt
from   matplotlib import cm


def centree(X) :
    ''' Xc = .centree(X);
    | Retourne Xc qui correspond au centrage des colonnes de X.
    '''    
    N, p = np.shape(X);
    meanx = np.mean(X,axis=0); # moyenne de chaque colonne
    repmx = nm.repmat(meanx,N,1); 
    Xc    = X - repmx; 
    return Xc; # X centrée

def centred (X,meanx=None,stdx=None,biais=1,coef=1) :
    ''' Xcr = .centred(X,biais);
    | Retourne Xcr qui correspond au centrage et à la réduction des colonnes
    | de le matrice (array) X.
    | Lorsqu'ils sont passés, meanx et stdx doivent etre de type array.
    | Si biais=1 (valeur par défaut), la réduction et débiaisée
    | (i.e. Normée par biais). coef est un coefficient multiplicateur.
    '''
    # On transforme les données sans dimension en tableau 1D
    # (pour prévenir des erreurs sur le type des données passées)
    if np.ndim(X) == 1 : # cas d'un vecteur
        X = np.reshape(X,(len(X),1));
    if (meanx is not None) and (np.ndim(meanx)==0) : # cas d'un scalaire
        meanx = np.array([meanx]);
    if (stdx is not None) and (np.ndim(stdx)==0) : # cas d'un scalaire
        stdx  = np.array([stdx]);

    # Suite ...
    N, p = np.shape(X); 
    if meanx is None :
        meanx = np.mean(X,axis=0);       # moyenne de chaque colonne
    if stdx is None :
        stdx  = np.std(X,axis=0,ddof=biais);  # std non biaisé de chaque colonne

    UN  = np.ones(np.shape(X));
    Moy = np.diag(meanx);
    Moy = np.dot(UN,Moy);
    Ecart_type = np.dot(UN,np.diag(stdx))
    Xcr = coef*(X - Moy) / Ecart_type;
    return Xcr; # X centré-réduit

def decentred (XN,crMoy,crStd,crCoef=1) :
    ''' X = .decentred(XN,biais);
    | Methode de denormalisation de données qui ont été centrees-reduites
    | avec centred. Il convient donc de passer les mêmes valeurs de
    | paramètre que ceux utilisées pour centred
    '''
    # On transforme les données sans dimension en tableau 1D
    # (pour prévenir des erreurs sur le type des données passées)
    if np.ndim(XN) == 1 : # cas d'un vecteur
        XN = np.reshape(XN,(len(XN),1));
    if np.ndim(crMoy) == 0 : # cas d'un scalaire
        crMoy = np.array([crMoy]);
    if np.ndim(crStd) == 0 : # cas d'un scalaire
        crStd = np.array([crStd]);

    # Suite ...
    UN  = np.ones(np.shape(XN));
    Moy = np.diag(crMoy);
    Moy = np.dot(UN,Moy);
    Ecart_type = np.dot(UN,np.diag(crStd))
    X = (XN * Ecart_type)/crCoef + Moy;
    return X


def normrange(X,a=0,b=1) :
    ''' Xr = normrange(X,a=0,b=1) :
    | Retourne Xr, LES vecteurs de la matrice X, normalisés dans
    | l'intervalle [a b]
    '''
    minX = np.min(X,0);
    maxX = np.max(X,0);
    dX   = maxX - minX;
    dr = b-a;
    Xr = a + dr * (X - minX) / dX;
    return Xr


def linreg (x, y) :
    ''' b0,b1,s,R2,sigb0,sigb1 = .linreg(x,y)
    | Calcule la régression linéaire de x par rapport a y.
    | En sortie :
    | b0, b1 : Coefficients de la droite de regression lineaire : y=b0+b1*x
    | s      : Erreur type de l'estimation
    | R2     : Coefficient de détermination
    | sigb0  : e.t(b0) : Ecart type  de la loi normal N(Beta0, sigb0) suivi par b0
    | sigb1  : e.t(b1) : Ecart type  de la loi normal N(Beta1, sigb1) suivi par b1
    '''
    N       = x.size;
    xmean   = np.mean(x);
    xmean2  = xmean*xmean;
    xcentre = x-xmean;
    ymean   = np.mean(y);
    ycentre = y-ymean;
    b1 = np.sum(ycentre*xcentre) / (np.sum(x*x) - N*xmean2);
    b0 = ymean - b1*xmean;  

    yc = b0 + b1*x;
    s  = np.sqrt(np.sum(pow((y-yc),2))/(N-2));
    R2 = np.sum(pow((yc-ymean),2))/np.sum(pow(ycentre,2));

    xcsq  = pow(xcentre,2);
    sigb0 = s*np.sqrt(1/N+xmean2/np.sum(xcsq));
    sigb1 = s/np.sqrt(np.sum(xcsq));
    
    return b0,b1,s,R2,sigb0,sigb1;


def errors(Yref,Yest,errtype=["rms"]) :
    ''' [Terr] = .errors(Yref,Yest,errtype)
    | Calcule de différents type d'erreur entre un vecteur de reference (Yref)
    | et un vesteur estimé (Yest). Ces types sont à passer dans le paramètre
    | errtype sous forme de chaine de caractères. Les types (et donc les erreurs)
    | possibles sont :
    | "errq"     : Erreur Quadratique 
    | "errqm"    : Erreur Quadratique Moyenne
    | "rms"      : RMS (Root Mean Squared Error)
    | "biais"    : Bias = moyenne des erreurs 
    | "errqrel"  : Erreur Quadratique relative
    | "errqrelm" : Erreur quadratique relative moyenne
    | "rmsrel"   : La RMS relative   
    | "biaisrel" : Le biais relatif   
    | "errrel"   : L'Erreur relative
    | exemple d'appel :
    |         rms, errqrel = .errors(Yref, Yest,["rms","errqrel"])
    '''
    nberr  = np.size(errtype);
    Nall   = np.size(Yref);
    Err    = Yref - Yest;   # Vecteurs des erreurs
    Errrel = Err / Yref;    # Vecteurs des erreurs relatives  
    
    TERR = np.zeros(nberr);
    for i in np.arange(nberr) :
        if errtype[i] == "errq" :           # Erreur Quadratique 
            TERR[i] = np.sum(Err**2);            
        elif errtype[i] == "errqm" :        # Erreur quadratique moyenne
            errq = np.sum(Err**2);
            TERR[i] = errq/Nall;            
        elif errtype[i] == "rms" :          # La RMS
            errq  = np.sum(Err**2);
            errqm = errq/Nall;
            TERR[i] = np.sqrt(errqm);           
        elif errtype[i] == "biais" :        # Bias = moyenne des erreurs
            TERR[i] = np.sum(Err)/Nall;            
        elif errtype[i] == "errqrel" :      # Erreur Quadratique relative
            TERR[i] = np.sum(Errrel**2);               
        elif errtype[i] == "errqrelm" :     # Erreur quadratique relative moyenne
            TERR[i] = np.sum(Errrel**2)/Nall;           
        elif errtype[i] == "rmsrel" :       # La RMS relative
            TERR[i] = np.sqrt(np.sum(Errrel**2)/Nall);            
        elif errtype[i] == "biaisrel" :     # Biais relatif
            TERR[i] = np.sum(Errrel)/Nall;            
        elif errtype[i] == "errrel" :       # L'Erreur relative
            TERR[i] = np.sum(abs(Errrel))/Nall; 
    #print("TERR=",TERR)
    return TERR


def matconf(Yref,Yest,visu=1) :
    ''' Mconf = .matconf(Yref,Yest,visu=1) :
    | Matrice de Confusion établie entre Yref et Yest.
    | En Entrée :
    | Yref et Yest : On considère 2 cas :
    |   cas 1 : Yref et Yest sont 2 vecteurs (i.e list d'éléments numériques
    |           uniquement) qui correspondent à des indices de classe. On convient
    |           que l'indice 0 correspond aux éléments non classés (classe 0).
    |   cas 2 : Yref et Yest sont des tableaux 2D (d'éléments numériques
    |           uniquement). Dans ce cas, on attribut à chaque ligne du tableau
    |           l'indice (à partir de 1) de son (1er) élément le plus grand qui
    |           correspondra alors à sa classe. On est ainsi rammené au cas 1.
    |   Yref  et Yest doivent avoir les mêmes dimensions.
    | visu : Si visu est différent de 0 alors la matrice de confusion est affichée.
    | En Sotie :
    | Mconf : La matrice de confusion.
    '''
    if np.shape(Yref) != np.shape(Yest) :
        print("Yref and Yest must be same dimensionnal");
        sys.exit(0);

    if np.ndim(Yref) == 1 :
        p = 1;
        Nelt = np.size(Yref);
    else : # p>1
        Nelt, p = np.shape(Yref);       
        Yref = np.argmax(Yref,1) + 1;
        Yest = np.argmax(Yest,1) + 1;

    Yref = Yref.astype(int); # On force à int
    Yest = Yest.astype(int); # On force à int

    Cmax     = np.max([np.max(Yref), np.max(Yest)]); # Classe max
    Nclasse  = Cmax + 1; # : + 1 pour tenir compte de la classe 0
    Mconf    = np.zeros((Nclasse,Nclasse),dtype=int); # init

    for i in np.arange(Nelt) :
        Mconf[Yref[i],Yest[i]] = Mconf[Yref[i],Yest[i]] + 1;

    if visu : # on affiche la matrice de confusion (par defaut)
        print("Predicted    ",end=''),
        for i in np.arange(Nclasse).astype(int) :
             print(i,"    ",end='');
        print("\nActual")
        for i in np.arange(Nclasse) :
            print(" % 3d       " %(i),end='')
            for j in np.arange(Nclasse) :
                 print("%3d   " %(Mconf[i,j]),end='');
            print("");
    return Mconf


def plotby2(X,Xclas=None,style=None,mks=10, cmap=cm.jet, \
            varnames=None,subp=1,Cvs=None,leg=None) :
    ''' plotby2(X,Xclas,style,varnames,subp,Cvs,leg)
    | Affichage de données (matrice X) par variables deux à deux,
    | Paramètres d'entrée :
    | X        : Matrice de données, variables en colonne, à ploter 2 à 2
    | Xclas    : Vecteur des classes des individus; par défaut, on considère 
    |            qu'il n'y en a qu'une.
    | style    : (Liste de chaines de 2 caractères) marqueurs et couleurs 
    |            à utiliser pour les différentes classes (à la manière de 
    |            plot : un caractère pour le marqueur et un pour la couleur)
    | mks      : Taille des marqueurs (MarKerSize)
    | cmap     : Si style n'est pas indiqué, on utilise une map de couleur, qui peut
    |            être précisée, pour l'affichage des différentes classes.
    | varnames : Noms des variables (utilisés pour les libellés des axes)
    | subp     : Si subp=1, affichage dans des subplots, sinon il y aura une
    |            figure pour chaque couple de variables.
    | Cvs      : [optionel] versus Xclas : classes de comparaison des individus.
    |            Si Cvs est passé, on entourera ceux qui n'ont pas la même 
    |            classe que dans Xclas.
    | leg      : On donne la possibilité d'afficher une légende pour chaque
    |            plot, mais en fait, ca peut ne pas être très pratique. il est
    |            préférable d'ajouter sois-même la légende au retour de la
    |            la fonction sur le dernier subplot (ou dernière figure), en 
    |            contrôlant les paramètres de la légende.
    '''
    plt.ion();
    #------------------------------------------------------
    N, p = np.shape(X);
    if Xclas is None :     # si pas de classe passée ... C'est comme 
        Xclas= np.ones(N); # si il n'y en avait qu'une
    nbCla  = max(Xclas);
    
    if np.size(Xclas) != N :
        print("plotby2: Xclas must be the same length as X");
        sys.exit(0);
    if varnames is not None :
        if np.size(varnames) != p :
            print("plotby2: the numbers of names in varnames must much the number of column in X");
            sys.exit(0);        
    if leg is not None :
        if np.size(leg) > nbCla :
            print("plotby2: warning: Too much legend given");
        if np.size(leg) < nbCla :
            print("plotby2: Not enought legend given (should be as much as classes)");
            sys.exit(0);

    # Si Cvs est passé, on entourera ceux qui n'ont pas la même classe
    if Cvs is not None :
        if np.size(Cvs) != np.size(Xclas) :
            print("plotby2: Xclas and Cvs must have the same length");
            sys.exit(0);            
        Idiff = np.where(Xclas != Cvs);

    # Gestion des couleurs par classe
    defaultstyle = 0;
    if style is not None :
        if np.size(style) != nbCla :
            print("plotby2: warning: when style is given, it must have as much element than the number of class");
            print("         turn to default map");
            defaultstyle = 1; # style par defaut
            Tcol = cmap(np.arange(1,256,round(256/nbCla)).astype(int)); # Tableau de couleurs par defaut
    else : # take the default
        defaultstyle = 1; # style par defaut
        Tcol = cmap(np.arange(1,256,round(256/nbCla)).astype(int)); # Tableau de couleurs par defaut

    # Pour gerer les subplot
    if subp==1 :
        nbsub  = p*(p-1)/2;
        nbsubc = np.ceil(np.sqrt(nbsub));
        nbsubl = np.ceil(nbsub/nbsubc);
        plt.figure();
        isub=1; 

    for i in np.arange(p-1) :
        for j in (np.arange(p-i-1)+i+1) :
            if subp==1 :
                plt.subplot(nbsubl,nbsubc,isub);
                isub = isub+1;
            else :
                plt.figure();
            
            for k in np.arange(nbCla).astype(int) :
                Icla = np.where(Xclas==k+1);
                if defaultstyle :
                    plt.plot(X[Icla,i][0], X[Icla,j][0],'.', markersize=mks, color=Tcol[k,:]);
                else :
                   plt.plot(X[Icla,i][0], X[Icla,j][0],style[k], markersize=mks);
            plt.axis("tight");
            
            if varnames is not None :  # Si des noms de variables sont passés on  
                plt.xlabel(varnames[i],fontsize=13); # les met en label des axes
                plt.ylabel(varnames[j],fontsize=13);

            # Si Cvs est passé, on entoure ceux qui n'ont pas la même classe
            if Cvs is not None :
               plt.plot(X[Idiff,i], X[Idiff,j],'ok',ms=8, \
                        fillstyle=None, mew=2, mec='k'); 
            #
            if leg is not None :
                plt.legend(leg);


def avotemaj(clA, votopt=0) :
    '''ELUE = avotemaj(clA, votopt)
    | Réalise un vote majoritaire parmi les classes passées
    | En entrée :
    | clA    : Un vecteur de N° classe (0 pour la classe nulle) :
    | votopt : Définit l'option en cas d'egalité du vote majoritaire :
    |          0 : En cas d'égalité de classe c'est alors la 1ère qui est retenue
    |          (c'est l'option par défaut), sinon, un tirage aléatoire est effectué.
    | En sortie :
    | ELUE  : La classe issue du scrutin.
    '''
    clA=clA.astype(int); # on force clA à être de type int
    # On va compter le nombre d'éléments pour chacune des classes
    Count = np.zeros(max(clA)+1).astype(int); #+1 pour une éventuelle classe nulle
    for i in np.arange(np.size(clA)) :
        Count[clA[i]] += 1;
    # Vote majoritaire :
    if votopt == 0 : # en cas d'égalité le 1er est avantagé
        ELUE = np.argmax(Count);
    else : # tirage aléatoire
        countmax = np.max(Count);                # décompte max
        imax     = np.where(Count==countmax)[0]; # indice de celles qui ont un décompte max
        nbmax    = np.size(imax);                # nombre de celles qui ont un décompte max
        theone   = np.floor(np.random.rand()*nbmax).astype(int); # tirage aléatoire
        ELUE     = imax[theone];
    return ELUE


def kvotemaj(icla, iprot, k=None, votopt=0) :
    ''' CPROT = kvotemaj(icla, iprot, k, votopt)
    | Vote majoritaire pour classer chacun des k prototypes (proto), dont les 
    | indices (iprot) sont associés aux classes des éléments (icla)
    | En entrée :
    | icla  : Les indices (vecteur) de N° de classe des données (0 pour une donnée
    |         non classée) :
    | iprot : Les indices (vecteur) des protos correspondant qui ont été associés
    |         aux données (par un algorithme de classification non supervisé
    |         (comme les k-moyennes par exemple), ou autre).
    |         icla et iprot doivent donc avoir la même taille.
    | k     : Le nombre de prototypes. Par défaut, on prend le max de iprot, mais
    |         il est préférable de passer explicitement le nombre k de protos pour
    |         éventuellement faire apparaitre ceux qui n'auraient rien capté.
    | votopt: Définit l'option en cas d'egalité du vote majoritaire :
    |         0 : En cas d'égalité de classe c'est alors la 1ère qui est retenue
    |         (c'est l'option par défaut), sinon, un tirage aléatoire est effectué.
    | Remarque : Les données doivent nécéssairement être associé à un proto ; par
    |         contre un proto qui n'aurait rien capté ne peut logiquement pas 
    |         apparaitre dans iprot.
    | En sortie
    | CPROT : Indices de classes attribués aux prototypes par vote majoritaire.  
    |
    | Exemple 1 :
    | icla  = np.array([1, 2, 2, 3, 3, 2, 0, 2, 0, 3, 1]);
    | iprot = np.array([1, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2]);
    | k=2
    | CPROT = [3 2] : ok le 1er proto aura la classe 3, le 2ème la classe 2
    |
    | Exemple 2 :
    | icla  = np.array([1, 2, 2, 3, 3, 2, 0, 2, 0, 3, 1]);
    | iprot = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]);
    | k=3
    | CPROT = [0 2 0] : ok le proto 2 aura la classe 2 le 1 et le 3 auront 
    |                                                      la classe nulle
    | Exemple 3 :
    | icla  = np.array([1, 2, 2, 3, 3, 2, 0, 2, 0, 3, 1]);
    | iprot = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
    | k = 2;
    | CPROT = [2 0] : ok le proto 1 à la classe 2 le proto 2 a la classe nulle
    '''
    if np.size(iprot) != np.size(icla) :
        print("votemaj: iprot must be the same length than icla");
        sys.exit(0);
    if k is None :
        k = max(iprot); # il serait préféreable de passer la valeur de k, mais bon
    if k < max(iprot) :
        print("votemaj: Warning : k is lower than max(iprot); max(iprot i tooken as k");
        k = max(iprot);
        
    CPROT = np.zeros(k).astype(int); # init of the resulting CPROT vote
    
    for j in np.arange(k) :
        Ij = np.where(iprot==j+1);  # Eléments associés au proto j(+1)
        if np.size(Ij)>0 :         
            # On fait un vote majoritaire en se servant de la classe connue des
            # éléments des données 
            CPROT[j] = avotemaj(icla[Ij],votopt=votopt)
        #else : Cela signifie que le proto j n'a rien capté ; il est
        #       affecté d'une "non classe"=0 de par l'init de classprot   
    return CPROT


def classperf(Yref, Yest, crit="max", miss=0, idgoodbad=0) :
    ''' PERF = .classperf(Yref, Yest, crit, miss)
    | Calcul la performance de classification entre des classes de
    | référence (Yref) et des classes estimées (Yest).
    | Chaque ligne de Yref et Yest correspondent à une classification.
    | Pour calculer la performance, on utilise le critère du max, c'est
    | à dire qu'on considère que la classification est correcte si les
    | max de 2 vecteurs (ligne) sont sur le même indice quelque soit l'écart
    | entre ces 2 max. Actuellement c'est le seul critère utilisé.
    | Le resultat PERF est une valeur comprise entre 0 et 1. le paramètre
    | miss, s'il est différent de 0, permet,d'obtenir plutôt une performance
    | de mal classé (dans ce cas on fait : PERF = 1-PERF, c'est tout).
    | idgoodbad est un paramètre qui a été rajouté pour pouvoir recupérer
    | en sortie, ou pas, les indices des biens et des mals classés.
    | Si idgoodbad==1 :
    |     PERF, idbad, idgood = .classperf(Yref, Yest, ...
    '''
    N, p = np.shape(Yref);
    M, q = np.shape(Yest);
    if N != M  or  p != q :
        print("classperf: Yref and Yest must have the same size");
        sys.exit(0);    
    a = np.argmax(Yref,1);
    b = np.argmax(Yest,1);
    idgood = np.where(a==b)[0]; # indices des biens classés
    idbad  = np.where(a!=b)[0]; # indices des mals classés
    PERF = np.size(idgood) / N;
    if miss==1 :
        PERF = 1 - PERF;
    if idgoodbad :
        return PERF, idbad, idgood;
    else :
        return PERF;


def shownpat2D(x,xshape,fr,to,subp=True,cmap=cm.jet,interpolation='none') :
    ''' .shownpat2D(x,xshape,fr,to,subp,cmap)
    | Affichage de vecteurs sous forme d'une image 2D
    | En entrée :
    | x      : La matrice des vecteurs ligne
    | xshape : [a, b] Les dimensions 2D à appliquer sur les vecteurs d'entrées 
    |          de l'ensemble d'apprentissage.  
    | fr, to : les indices (from,to) des vecteurs à prendre dans x.
    | subp   : Affichage en subplot (si True) ou pas, auquel cas il y aura une
    |          figure par vecteur ligne selectionnés.
    | cmap   : La map de couleur à utiliser.
    | interpolation : Valeur du paramètre d'interpolation pour la fonction 
    |          imshow de Matplotlib.
    '''
    plt.ion();
    if subp :
        plt.figure();
        nbpat = to-fr+1;
        subpa = np.ceil(np.sqrt(nbpat));
        subpb = np.ceil(nbpat/subpa);
    for k in np.arange(nbpat) :
        pat = np.reshape(x[fr+k-1,:],(16,16));
        pat = np.reshape(x[fr+k-1,:],xshape);
        if subp :
            plt.subplot(subpb,subpa,k+1);
        else :
            plt.figure();
        plt.imshow(pat, cmap=cmap,interpolation=interpolation);
    plt.show();


def rota2d(Z,theta=0) :
    ''' Y = .rota2d(Z,theta) : Rotation d'une matrice Z comportant N lignes et deux
    | colonnes (Nx2) d'un angle de theta degrés compris dans le sens contraire aux
    | aiguilles d'une montre. Y en sortie est la matrice résultante de la rotation
    | de Z.
    '''
    A = np.pi * theta / 180; # passage des degrés aux radians
    # Constitution de la matrice de rotation 2D
    cos_t = np.cos(A);    sin_t = np.sin(A);   
    R = np.array([[cos_t, sin_t], [-sin_t, cos_t]]);
    # Rotation
    Y = np.dot(Z,R);
    return Y


def gen2dstd(n,M=[0,0],E=[1,1],theta=0) :
    '''
    | GEN2DSTD :         Crée une matrice 2D (i.e. 2 colonnes)
    |    de données suivant une loi gaussienne en fonction d'une 
    |    moyenne, d'un écart type et selon une rotation.
    |    
    | X = gen2dstd(n,M,E,theta);
    |
    | En entree : 
    | n     : Nombre de points
    | M     : Moyennes de chacune des 2dimensions. defaut=[0, 0].
    | E     : Ecarts types de chacune des 2 dimensions. defaut=[0, 0].
    | theta : Angle de rotation dans le sens inverse des aiguilles d'une montre.
    |         defaut=0.        
    '''
    # Tirage aléatoire des données selon une loi normale N(M,E).
    X = np.array([np.random.randn(n,1)*E[0], np.random.randn(n,1)*E[1]]).T;
    X = X.reshape(n,2);
    # Tourner la matrice de 'theta' degrés dans le sens contraire aux aiguilles d'une montre
    X = rota2d(X,0);  
    # Déplacer la matrice de M
    X = np.array([X[:,0]+ M[0],  X[:,1]+ M[1]]).T;
    return X

def gen2duni (Ndata,a,b,theta=0) :
    '''
    | GEN2DUNI :         Génération d'un ensemble de données 2D,  
    | de taille Ndata X 2 selon une loi uniforme dans l'intervalle 
    | de surface rectangle [a b] : U(a,b) (a et b étant des vecteurs 
    | de dimension 2)
    |    
    | X = gen2duni(Ndata,a,b,[theta]);
    |
    | En entree : 
    | Ndata : Nombre de points
    | a, b  : paramètres de la loi uniforme : vecteurs des borne inf 
    |         et sup des l'intervalles de la surface en x et y : 
    |         a=[ax, ay] correspond au coin inférieur gauche,
    |         b=[bx, by] correspond au coin supérieur droit, 
    | theta : Angle de rotation dans le sens inverse des aiguilles d'une montre.
    |         defaut=0.        
    '''
    Zx = np.random.rand(Ndata);     # uniforme sur [0 1]
    Zx = a[0] + Zx*(b[0]-a[0]);     # uniforme sur [0+a  (b-a)+a] => sur [a b]
    Zy = np.random.rand(Ndata);     # uniforme sur [0 1]
    Zy = a[1] + Zy*(b[1]-a[1]);     # uniforme sur [0+a  (b-a)+a] => sur [a b]
    Z  = np.transpose([Zx, Zy]); #print(np.shape(Z)); (167,2)
    #
    # Fait tourner la matrice generee par 'theta' degrees dans le sens 
    # contraire aux aiguilles d'une montre
    #X= Z * Rota2D(theta);
    A = np.pi * theta / 180;
    cos_t = np.cos(A);    sin_t = np.sin(A);
    R = np.array([ [cos_t, sin_t], [-sin_t, cos_t] ]); #print(R)
    X = np.dot(Z,R); #print(np.shape(X));
    #    
    return X

#-----------------------------------------------------------------
#===================== miscellaneous section =====================
def intarray2str(intarray) :
    ''' STRARRAY = intarray2str(intarray)
    | Traduit un tableau d'entier en tableau de chaine de caractères
    | (Parce que j'ai pas encore trouver la fonction qui le fait ...)
    '''
    STRARRAY=[]
    for i in np.arange(np.size(intarray)) :
        STRARRAY.append(str(intarray[i].astype(int)));
    return STRARRAY

def vdist(X,Y) :
    lx, cx = np.shape(X);
    ly, cy = np.shape(Y);
    if lx != ly :
       print('vdist : error : X et Y doivent avoir le meme nombre de lignes')
       sys.exit(0);
    Dist = np.zeros((cy,cx))   
    for i in np.arange(cx) :
        for j in np.arange(cy) :
            Dist[j,i] = np.sqrt(sum((X[:,i]-Y[:,j] )**2));
    return Dist

#-----------------------------------------------------------------
#============= more or less matlab like section ==================
def unique(A) :
    ''' U = unique(A)
    | Retourne un tableau U contenant les uniques occurrences des 
    | différentes lignes du tableau A. C'est à dire, si l'on préfère,
    | les occurrences des lignes de A mais sans répétition.
    '''
    b = np.ascontiguousarray(A).view(np.dtype((np.void, A.dtype.itemsize * A.shape[1])))
    U = np.unique(b).view(A.dtype).reshape(-1, A.shape[1]);
    return U;

def ismember(A, B) :
    ''' FLAG = ismember(A, B)
    | Retourne un vecteur (FLAG) de la taille de A,dont chaque
    | ième composante est valoriées à 1 si le ième élément (c'est
    | à dire la ième ligne) du tableau A est égal au vecteur B, et
    | à 0 sinon.
    | A et B sont de type numérique; B ne correspond qu'à un seul
    | vecteur (ces 2 derniers points constituent une différence
    | importante avec la fonction unique de matlab)
    '''
    Count  = np.array([ np.sum(a == B) for a in A ]);
    n, p   = np.shape(A);
    FLAG   = np.zeros(n);
    member = np.where(Count==p);
    FLAG[member] = 1;
    return FLAG;

def unifpdf(x,a,b) :
    ''' p = unifpdf(x,a,b)
    | Fonction de densité de la loi Uniforme sur l'intervalle a b (U(a,b)
    | pour l'ensemble des valeur de x.
    '''
    p = np.zeros(x.size);
    I = np.where((x>=a) & (x<=b))
    p[I] = 1/(b-a);
    return p
    

#-----------------------------------------------------------------
#========================= cstool section ========================
def concstrlist (L1,L2) : # de meme taille
    ''' Concatenation terme à terme de 2 listes de string de meme
    taille (en attendant de trouver la fonction python qui le fait)
    '''
    if len(L1) != len(L2) :
        print("triedtools.concstrlist : les 2 listes doivent avoir la meme taille");
        sys.exit(0)
    LL=[]
    for i in np.arange(len(L1)) :
        LL.append(L1[i]+L2[i])
    return LL
def tprin(T,fmt) :
    dim = np.ndim(T)
    if dim == 1 :
        nbe = np.size(T,0)
        for i in range(nbe) :
            print(fmt % (T[i]), end="");
        print('');
    else : # on considère tableau 2D
        nbl, nbc = np.shape(T);
        for i in range(nbl) :
            for j in range(nbc) :
                print(fmt % (T[i,j]), end="");
            print('');

def klavier(banner=None):
    import code
    #http://vjethava.blogspot.fr/2010/11/matlabs-keyboard-command-in-python.html
    ''' Function that mimics the matlab keyboard command '''
    # use exception trick to pick up the current frame
    try:
        raise None
    except:
        frame = sys.exc_info()[2].tb_frame.f_back
    print("# Use quit() to exit :) Happy debugging!")
    # evaluate commands in current namespace
    namespace = frame.f_globals.copy()
    namespace.update(frame.f_locals)
    try:
        code.interact(banner=banner, local=namespace)
    except SystemExit:
        return 

#-----------------------------------------------------------------
#=================================================================
#
