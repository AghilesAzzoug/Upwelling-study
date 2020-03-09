def trieddeep():
    """
    Méthodes de deep learning
    PMC (Perceptron Multi-Couches) : 
        pmcinit   : Initialisation des poids
        pmcout    : Calcule de la passe avant
        pmctrain  : Algorithme batch de convergence
    Réseau de Neurones à masque et poids partagés (réseau de convolution)
        wsminit   : Initialisation des poids
        wsmout    : Calcule de la passe avant
        wsmtrain  : Algorithme de convergence
        wsmshowccpat : Affichage de l'activation de la carte des caracteristiques
    """
    return None

import sys
import numpy as np
import matplotlib.pyplot as plt
from   matplotlib import cm
import copy
from   triedpy import triedtools as tls

#============================================================================
#============================================================================
#                          Perceptron Multi-Couches
#============================================================================
#============================================================================    
def pmcinit(Xi,Yi,m=[],k=0.1) :
    '''WW = pmcinit(Xi,Yi,m,k) : Initialisation des matrices de poids
    | d'un PMC dimensionnées selon les matrices d'entrées et de sorties
    | du PMC et des nombre de neurones cachés;
    | En entrees :
    | Xi : (matrix 2D) Ensemble des données d'apprentissage en entrée du PMC
    | Yi : (2d matrix) Ensemble des données d'apprentissage en sortie du PMC
    |      (Sorties désirées) 
    | m  : (vecteur) Nombre de neurones sur les couches cachées. Le nombre de
    |       valeurs détermine le nombre de couches cachées
    | k  : (scalar) Facteur multiplicatif initial appliqué sur les poids.
    |      Valeur par defaut = 0.1 
    | En sortie :     
    | WW : Les matrices de poids initiaux.
    '''    
    m0 = np.size(Xi,1);
    mq = np.size(Yi,1);
    m  = np.insert(m, 0, m0); 
    m  = np.append(m, mq);
    m  = m.astype(int); # forcage à int
    if np.ndim(m) != 1 :
        print("pmcinit: m must be one dimension");
        sys.exit(0);
    nbc = np.size(m); # nombre total de couches
    WW = [];
    for i in np.arange(nbc-1) :
        W  = k*np.random.randn(m[i]+1,m[i+1]);
        WW.append(W);
    WW[nbc-2][m[nbc-2],:] = np.mean(Yi,0); # quid other couches ?
    # Not Done : level test on input and output data
    return WW 

#=================================================================
def pmcout(X,WW,FF,fparm=[0.6667, 1.7159, 0.0]) :
    ''' Y = pmcout(X,WW,FF,fsigparm) : Calcul des valeurs de sortie d'un PMC
    | En entrée :
    | X   : (matrice) Données d'entrées du PMC
    | WW  : liste des matrices de poids pour chaque couche (ci->ci+1)
    | FF  : liste de string des fonctions d'activation des neurones cachés.
    |       Les valeurs possibles sont :
    |       'tah' : pour la tangente hyperbolique 
    |       'sig' : pour la fonction sigmoide
    |       'lin' : pour la fonction linéaire
    |       'exp' : pour la fonction exponnentielle
    |       Il doit y avoir autant de fonctions dans FF que de matrice de poids
    |       dans WW. 
    | fsigparm : paramètres de la fonction sigmoide : [asympt, pente, offset]. 
    |            Valeurs par défaut: [0.6667, 1.7159, 0.0]).
    | En sorties :
    | Y  : Sorties du PMC pour l'ensemble X.
    '''
    nbc = len(WW); # nombre de couches sans l'input
    if np.size(FF) != nbc :
        print("pmcout: Le nombre de fonctions de transfert doit correspondre au nombre de couche à calculer");
        sys.exit(0);
    # Init 1 fois parm sigmoide, tant pis si on en a pas besoin
    if FF.count('sig') > 0 :
        asympt = fparm[0];
        pente  = fparm[1];
        offset = fparm[2];

    onell = np.ones((np.size(X,0),1)); # biais
    
    for i in np.arange(nbc) :  
        X    = np.concatenate((X, onell),axis=1);  # Ajout du biais aux entrées
        Ai    = np.dot(X,WW[i]);                   # Somme pondérée des entrées
        if   FF[i]=="tah" :
            X = np.tanh(Ai)
        elif FF[i]=="sig" :
            ekx = np.exp(-pente*Ai);
            X = asympt*(1-ekx)/(1+ekx) + offset;
        elif FF[i]=="lin" :
            X = Ai;
        elif FF[i]=="exp" :
            X = np.exp(Ai);       
        else :
            print("pmcout: Unknown activation function %d." %(i+1));
            sys.exit(0);
    #return Xi; # Les premiers seront les derniers
    Y = X;      # Les premiers seront les derniers
    return Y; 

#=================================================================
def pmctrain(Xa,Ya,WW,FF,nbitemax=1000, fparm=[0.6667,1.7159,0.0], \
             gstep=0.1, alpha=0.4737, weivar_seuil=10**(-6), \
             tfp=10,dfreq=-1,dprint=1,dcurves=0, dperf=0, \
             Xv=None,Yv=None,pval=0, ntfpamin=100,dval=1) :
    ''' Sans données de validation : WW = pmctrain(Xa,Ya,WW,FF,...)
    | Avec données de validation   : WW, it_minval = pmctrain(Xa,Ya,WW,FF,...)
    | Effectue les itérations d'apprentissage d'un PMC par l'algorithme de
    | rétropropagation de gradient dans sa version batch.
    |
    | En Entrée :
    | Xa : (matrice) Données d'apprentissage en entrée du PMC
    | Ya : (matrice) Sorties désirées (données de référence) associées à Xa.
    | WW  : liste des matrices de poids pour chaque couche (ci->ci+1)
    | FF  : liste de string des fonctions d'activation des neurones cachés.
    |       Les valeurs possibles sont :
    |       'tah' : pour la tangente hyperbolique 
    |       'sig' : pour la fonction sigmoide
    |       'lin' : pour la fonction linéaire
    |       'exp' : pour la fonction exponnentielle
    |       Il doit y avoir autant de fonctions dans FF que de matrice de poids
    |       dans WW. 
    | nbitemax : (entier) Nombre maximum d'itérations (defaut: 1000)
    | fsigparm : (vecteur) Paramètres de la fonction sigmoide :
    |            [asympt, pente, offset], (defaut: [0.6667, 1.7159, 0.0]).
    | gstep    : (réel) Pas de gradient initial (defaut: 0.1) 
    | alpha    : (réel) facteur de momentum (souvenir des états passés)
    |            default value=0.9/(1+.9)=0.4737)    
    | weivar_seuil : (réel) Paramètres pour l'arret sur un seuil minimum de 
    |            variation des poids. Valeur par défaut = 10**(-6).
    | tfp      : (entier) Top Frequency Processing : Permet de diminuer le nombre 
    |            de certains calcul (eraly stopping) et de stockage en ne les
    |            effectuant que tous les tfp fois. Ne doit etre supérieur à
    |            dfreq ou nbitemax (defaut: 10)
    | dfreq    : (entier) Fréquence d'affichage. par defaut, qu'un affichage final.    
    | dprint   : (entier) print (ou pas) des valeurs pendant le déroulement de    
    |            l'apprentissage, aussi en fonction des paramètre dperf, dval qui
    |            suivent. Valeur par défaut = 1 : affichage. 
    | dcurves  : Affichage (ou pas) des courbes du déroulement de l'apprentissage
    |            aussi en fonction des paramètre dperf, dval qui suivent.
    |            valeur par défaut = 0 : pas d'affichage.
    |            >= 1 : Affichage
    |               2 : Les erreurs sont affichées en log.
    | dperf    : Affichage (si=1) ou pas (si=0) de valeurs associées aux performances
    |            qui se comprennent ici comme des pourcentages d'erreur.
    | Xv       : (matrice) Données d'entrée de validation. Permet de pratiquer
    |            l'arrêt prématuré.
    | Yv       : (matrice) Sorties désirées associées aux données de validation Xv.
    | pval     : Indique sur quelle valeur doit se faire l'appréciation de l'arrêt
    |            prématuré :
    |            0 : sur l'erreur en validation (c'est le cas pas défaut)
    |            1 : sur la performance en validation 
    | ntfpamin : (entier) Nombre maximum de tfp à faire après le dernier minimum 
    |             constaté en validation (défaut: 100).
    | dval     : Affichge (si=1) ou pas (si=0) des valeurs qui se concerne l'ensemble
    |            de validation
    |
    | En Sortie :
    | Sans données de validation :
    |    WW : Matrices des poids obtenus à la fin de la procédure
    | Avec données de validation   :
    |    WW : Matrices des poids obtenus au minimum sur l'ensemble de validation.
    |         (Les poids de fin de procédure sont stockés dans un fichier: pmcendwei)
    |    it_minval: l'indice de l'itération où le minimum en validation a été trouvé.
    '''
    nbitamin = ntfpamin; #...
    nbc = len(WW); # nombre de couches sans l'input
    if np.size(FF) != nbc :
        print("pmctrain: Le nombre de fonctions de transfert doit correspondre au nombre de couches à calculer");
        sys.exit(0);
    if dfreq==-1 :          
        dfreq = nbitemax;
    if tfp>dfreq or tfp>nbitemax :
        print("pmctrain: tfp parameter must not be greater than dfreq or nbitemax");
        sys.exit(0);
                            
    asympt = fparm[0];
    pente  = fparm[1];
    offset = fparm[2];
    #
    #Initialisations ------------------------
    # Dimension stuff
    ell   = np.size(Xa,0);
    if np.size(Ya,0) != ell :
        print("pmctrain: Xa and Ya must have the same number of line");
        sys.exit(0);
    onell = np.ones((ell,1)); # biais...
    t=[];
    for i in np.arange(nbc) :
        t.append(np.size(WW[i],0));

    # Pour l'ajustement des pas (1)
    a     = 1.5;
    b     = 1/a;
    dim   = 0.5; # diminution du pas en cas d'augmentation de l'erreur

    # Init sauvegardes pour marche arriere and other
    #WWp    = np.copy(WW);
    WWp     = copy.deepcopy(WW);
    gradWWp = []; 
    OVF     = [];
    WWpp    = [];
    sizWW   = [];
    nbWW    = [];
    oneWW   = [];
    PAS     = [];
    for i in np.arange(nbc) :
        gwpi    = np.zeros(WW[i].shape);
        gradWWp.append(gwpi);
        # Pour l'ajustement des pas (2)
        ovfi    = 1e3 * np.ones(np.shape(WW[i]));  
        OVF.append(ovfi);
        # For weights variation threshold check and deal (1)
        wppi  = np.zeros(np.shape(WW[i]));
        WWpp.append(wppi);
        sizwi = np.shape(WW[i]);
        sizWW.append(sizwi);      
        nbwi  = np.prod(sizwi);
        nbWW.append(nbwi);       
        onewi = np.ones(nbwi);
        oneWW.append(onewi);
        # ...
        pasi = gstep * np.ones(sizwi)*2/ell;
        PAS.append(pasi);

    gradWW = copy.deepcopy(gradWWp); #np.copy(gradWWp); # easier if initialized
    descWW = copy.deepcopy(gradWW);  #np.copy(gradWW);  # easier if initialized
    errold = 1e16;

    # For weights variation threshold check and deal (2)
    sbougeti="";            
    #lambada = 0;               # Weight decay factor
    #lambda  = 0 #=lambada.*ones(size(w2));
    errtot  = []; 	        # memoire erreur pour sortie
    perftot = []; 	        # memoire performance pour sortie
    nballit = 0;                # nombre total d'iterations effectuees
    nbite   = 0;                # nombre de 'bonnes iterations'
    ctrback = 0;                # compteur d'iteration de marche arriere
    asympt=0.6667; pente=1.7159; offset=0.0;    # sigmoide parameters
    weivar_freq = 200;      # check weights variation frequency default value (2nd stop criterium)

    # Do we use Validation data
    valid_on = False; 
    if Xv is not None and Yv is not None :
        if np.size(Xv,0) != np.size(Yv,0) :
            print("pmctrain: Xv and Yv must have the same number of line");
            sys.exit(0);
        if np.size(Xv,0) > 0 :
            valid_on = True;
            cpit_addmin = 0;    # ComPtage du nbre d'IT apres le Min
            minval      = 1e16; # Initialisation du minimum en validation
            WWMV        = WW;   # garder les poids du min en validation
            it_minval   =   0;  # Iteration du Min en sortie
            errtotval   =  [];  # memoire erreur pour sortie
            perftotval  =  [];  # memoire performance pour sortie
    if valid_on == False :
        dval = 0;   # Si dval était à 1, on le force à 0.

    #--------------------------------------------------------
    if (dcurves>0) :
        plt.figure();
        plt.ion();

    #=========================================================
    # boucle principale d'Apprentissage
    continumongars = 1;
    while (nballit < nbitemax) & continumongars :

        #Increment d'iterations .......................... 
        nballit = nballit+1;
        
        # Propagation Avant -------------------------------
        Yi = [];
        Xi = np.copy(Xa);
        for i in np.arange(nbc) :  
            Xi    = np.concatenate((Xi, onell),axis=1);  #Ajout du biais aux entrées
            Ai    = np.dot(Xi, WW[i]);                   # Somme pondérées des entrées
            if   FF[i]=="tah" :
                Xi = np.tanh(Ai);
            elif FF[i]=="sig" :
                ekx = np.exp(-pente*Ai);
                Xi = asympt*(1-ekx)/(1+ekx) + offset;
            elif FF[i]=="lin" :
                Xi = Ai;
            elif FF[i]=="exp" :
                Xi = np.exp(Ai);       
            else :
                print("pmctrain: Unknown activation function %d." %(i+1));
                sys.exit(0);
            Yi.append(Xi);    

        # Coût --------------------------------------------
        err =  (Yi[nbc-1] - Ya);
        errnew = np.sum(err*err) #+ sum(sum(lambda.*w2.^2));

        # Top Frequency Processing ------------------------
        # (to avoid too much computation and storage)
        if np.remainder(nballit,tfp)==0 :
            # on App
            errtot.append(errnew); # Sto Eapp
            if dperf  : #si cas de classification
                perfapp = tls.classperf(Ya,Yi[nbc-1],miss=1);  # Compute Papp
                perftot.append(perfapp);                # Sto Papp
                
            # on Validation (if given) . . . . . . . 
            if valid_on :
                Yvi = pmcout(Xv,WW,FF,fparm=fparm);                
                if pval==1 : # C'est la perf qui doit servir pour l'early stop.
                    perfval = tls.classperf(Yv,Yvi,miss=1);
                    earlyval = perfval;                    
                else   :    # c'est l'erreur quadratique
                    errqval  = np.sum((Yv-Yvi)**2);
                    earlyval = errqval;
                
                if dval : # On veut afficher les valeurs de validation ;
                          # il faut calculer ce qui ne l'a pas été
                    if pval==0 : 
                        perfval = tls.classperf(Yv,Yvi,miss=1);
                    else :
                        errqval = np.sum((Yv-Yvi)**2);                       
                    errtotval.append(errqval);
                    perftotval.append(perfval);

                # Early stopping (stop criterium 1)
                if earlyval < minval :  # detection du minimum en validation
                    minval      = np.copy(earlyval); 
                    it_minval   = np.copy(nballit);
                    cpit_addmin = 0;
                    WWMV = copy.deepcopy(WW); #np.copy(WW);
                           # garder les poids du min en validation. MV stand for Min Validation             
                else :
                    cpit_addmin += 1;
                    if cpit_addmin >= nbitamin :
                        print("pmctrain: Min Val (%f) at it %d then stop %d it after" %(minval,it_minval,nbitamin*tfp));
                        continumongars=0;         
        
        # Marche arriere ----------------------------------
        if errnew > errold :
            ctrback = ctrback+1; # Increment des mauvaises iterations
            WW      = copy.deepcopy(WWp);
            gradWW  = copy.deepcopy(gradWWp);
            descWW  = copy.deepcopy(gradWW);
            for i in np.arange(nbc) :
                PAS[i] = PAS[i] * dim;
                WW[i]  = WW[i] - PAS[i] * gradWW[i];
            #       
        else :    # Ca marche => un pas en avant :
            nbite = nbite + 1;       # Increment des bonnes iterations
        
            # Back Propagation -------------------------------------
            # From la 1ère couche de sortie
            dJdy = 2*err;
            y = Yi[nbc-1];
            if   FF[nbc-1]=="tah" :
                f2Ai = (1-y*y);
            elif FF[nbc-1]=="sig" :
                f2Ai = asympt * (1-y*y);        
            elif FF[nbc-1]=="lin" :
                f2Ai = 1;  
            elif FF[nbc-1]=="exp" :
                f2Ai = y;
            #else : Ne devrait pas se produire : déjà  controlé lors de passe avant
            dJdAi = dJdy * f2Ai;
            if nbc > 1 :
                Xai    = np.concatenate((Yi[nbc-2], onell),axis=1);
            else :
                Xai    = np.concatenate((Xa, onell),axis=1);           
            gradWW[nbc-1] = np.dot(Xai.T,dJdAi) / ell; #+ (2*lambda.*w2))./ell;
            descWW[nbc-1] = (1-alpha)*gradWW[nbc-1]  +  alpha*descWW[nbc-1];

            # Les couches cachées internes      
            for i in np.arange(nbc-1) :
                Wi = WW[nbc-1-i];
                ti = t[nbc-1-i];
                Wi = Wi[0:ti-1,:];
                y = Yi[nbc-2-i];
                if   FF[nbc-2-i]=="tah" :
                    dJdAi = np.dot(Wi,dJdAi.T).T * (1-y*y);
                elif FF[nbc-2-i]=="sig" :
                    dJdAi = np.dot(Wi,dJdAi.T).T * asympt * (1-y*y);        
                elif FF[nbc-2-i]=="lin" :
                    dJdAi = np.dot(Wi,dJdAi.T).T;  
                elif FF[nbc-2-i]=="exp" :
                    dJdAi = np.dot(Wi,dJdAi.T).T * y;
                #else : Ne devrait pas se produire : déjà  controlé lors de passe avant
                if i < nbc-2 : # pas fin
                    Xai    = np.concatenate((Yi[nbc-3-i], onell),axis=1);
                else :
                    Xai    = np.concatenate((Xa, onell),axis=1);
                gradWW[nbc-2-i] = np.dot(Xai.T,dJdAi) / ell;
                descWW[nbc-2-i] = (1-alpha) * gradWW[nbc-2-i] + alpha * descWW[nbc-2-i];
            
            # sauvegardes (avant correction des poids) ............ 
            errold  = errnew;			
            WWp     = copy.deepcopy(WW);
            gradWWp = copy.deepcopy(gradWW);

            # Ajustement des pas et correction des poids
            for i in np.arange(nbc) :
                # Ajustement des pas ..................................
                test   = (gradWW[i] * descWW[i]) >= 0;
                PAS[i] = (test*a + np.invert(test)*b) * PAS[i];
                PAS[i] = (PAS[i]<=OVF[i])*PAS[i] + (PAS[i]>OVF[i])*OVF[i];           
                # + voir aussi ci-dessous
     
                #Correction des poids (apres ajustement des pas) -----
                WW[i] = WW[i] -  PAS[i] * descWW[i];

            # Check weights variation (stop criterium 2) ---------
            if np.remainder(nbite,weivar_freq)==0 :
                bougeti = 0.0;
                for i in np.arange(nbc-1) :
                    Wi       = WW[i];
                    abswi    = abs(Wi);
                    abswi    = np.reshape(abswi,nbWW[i]);
                    maxwione = np.max((oneWW[i],abswi),axis=0);
                    maxwione = np.reshape(maxwione,sizWW[i]);
                    maxi     = max(np.max(abs(WWpp[i]-Wi)/maxwione,axis=0))
                    if maxi > bougeti :
                        bougeti = maxi;
                if bougeti < weivar_seuil :  # est-ce le bon critère d'arret : les poids peuvent bouger
                    continumongars=0;        # relativement à leur propre valeur sans que cela n'impact 
                    print('pmctrain: Weight variation threshold stop\n'); # plus la fonction de cout significativement !!!???
                                                                     # (nécessite de plus la sauvegarde des matrices)
                sbougeti = "%6.6f" %(bougeti);                    
                
            # Régulierement, tous les 1000 it (arbitrairement?) on se repositionnne et on redistribue
            # le jeu ... (sachant qu'avec d'autres valeurs (100, 2000) ca fait une différence)
            if np.remainder(nbite,1000)==0 : 
                # Sauvegarde des Poids (pour apprecier leur variation (cf bougeti) entre 1000 !? itérations)
                WWpp = np.copy(WW); 
                # Moyennage des Pas (pour redistribuer le jeu)
                for i in np.arange(nbc) :
                    PAS[i] = np.mean(PAS[i]) * np.ones(sizWW[i])*2/ell;  

            # fin back propagation--------------------------------        

        # Affichage ----------------------------------------------
        if  (dprint>0 or dcurves>0) \
        and (np.remainder(nballit,dfreq)==0 or continumongars==0 or nballit>=nbitemax) :

            # D'abord les prints (si demandés)
            if dprint :
                # L'entête
                if np.remainder(nballit,20*dfreq)==dfreq :
                    print("|    #epoch  |  Errorapp  ", end='');
                    if dval :
                        print("|   ErrqVal  ", end='');
                    if dperf :
                        print("| PerfApp ", end='');
                        if dval :
                            print("| Perfval ", end='');
                    print("| bougeti  |");
               
                # Les valeurs
                print("|%5d %5d | %10.6f " %(nballit,nbite,errnew), end='');
                if dval :
                    print("| %10.6f " %(errqval), end='');              
                if dperf :         
                    print("|  %.4f " %(perfapp), end='');           
                    if dval :
                        print("|  %.4f " %(perfval), end='');                   
                print("| %8s |" %sbougeti);
            
            # Ensuite la figure des courbes (si demandée)
            if dcurves > 0 : # Si courbes demandée(s) 
                absci = np.arange(np.size(errtot))+1; # abscisse pour les plots
                # Plot des perf (si y'en a)
                if np.size(perftot) > 0 : # Si y'a des perf                    
                    plt.subplot(2,1,2);
                    plt.plot(absci,perftot,'.-b');
                    if dval :
                        plt.plot(absci, perftotval,'.-r');
                    plt.subplot(2,1,1); # pour les erreurs (rem: pas de perf => pas de subplot)
                # Plot des erreurs
                if dcurves == 1 :
                    plt.plot(absci,errtot,'.-b');
                else :
                    plt.plot(absci,np.log(errtot),'.-b');
                if dval :
                    if dcurves == 1 :
                        plt.plot(absci,errtotval,'.-r');
                    else :
                        plt.plot(absci,np.log(errtotval),'.-r');
                plt.axis("tight");
                plt.draw();
                    
    # fin du while : fin de la boucle principale d'Apprentissage
    #=============================================================
    # Last Actions --------------------------------------------
    print("ctrback=",ctrback);
    print("pmctrain: %d iterations done" %nballit);
    #
    if dcurves > 0 : # Ornemantation à la fin seulement
        plt.xlabel("x %d Epoch" %tfp);
        if np.size(perftot) > 0 : # Si y'a des perf 
            plt.subplot(2,1,2);
            plt.ylabel("Perf: % Err. de classif.");
            plt.axis("tight");
            plt.subplot(2,1,1); # pour les erreurs (rem: pas de perf => pas de subplot)
        if dcurves == 1 :
            plt.ylabel("$\sum$ Errors Quadratiques");
        else :
            plt.ylabel("log($\sum$ Errors Quadratiques)");
        plt.axis("tight");
        #if valid_on :
        if dval :
            plt.legend(["Learning","Validating"]);
        else :
            plt.legend(["Learning"]);
        plt.show();

    # Last computations for return values
    if valid_on :
        np.save("pmcendwei", WWp); # Sauvegarde des poids en fin d'apprentissage  
        return WWMV, it_minval   
    else :
        return WWp


#============================================================================
#============================================================================
#                          Masque et Poids partagés
# les méthodes wsm_ sont à usage interne, et ne sont donc pas nécessairement
# documentées.
#============================================================================
#============================================================================
def wsm_subarr (X,a,lig,col) :
    n, p = np.shape(a);
    for i in np.arange(n) :
        for j in np.arange(p) :
            X[lig+i,col+j] = a[i,j];
    return X;

def wsm_convmask(xshape,mask,skip) :
    n, p  =  xshape;
    m, q  =  mask;
    skipl, skipc = skip;
    # h_size, sachant que des bords sont rajoutés ...:
    h_lig  = np.floor((n+m-1)/skipl).astype(int);
    h_col  = np.floor((p+q-1)/skipc).astype(int);
    h_size = [h_lig, h_col];
    loopl = np.arange(1,h_lig*skipl+1,skipl)-1
    loopc = np.arange(1,h_col*skipc+1,skipc)-1
    loop  = [loopl, loopc]  
    return h_size, loop

#=================================================================
def wsminit(xshape,mask,skip,dimout) :
    ''' w1,b1,w2,b2 = wsminit(xshape,mask,skip,dimout)
    | Initialisation des poids (pour l'architecture particulière indiquée dans
    | la méthode wsmtrain).
    | En Entrée
    | xshape : Les dimensions 2D à appliquer sur les vecteurs d'entrées de
    |          l'ensemble d'apprentissage.
    | mask   : [a, b] : Taille du masque de convolution (à appliquer sur les
    |          entrées 2D).
    | skip   : [l, c] : Décalage à appliquer pour le masque (mask) d'entrée.
    |           l : décalage horizontal (en nombre de ligne1)
    |           c : décalage vertical (en nombre de colonne)
    | dimout : Dimension des vecteurs de sortie de l'ensemble d'apprentissage.
    | En Sortie :
    | w1,b1,w2,b2 : Poids et seuils initiaux.
    '''
    m, q = mask;
    if skip[0] > m or skip[1] > q :
        print("wsm_init: Warning : Convolution non recouvrante");
    
    h_size, loop  = wsm_convmask(xshape,mask,skip); 
    w1 = (2*np.random.rand(m , q)-1)   / np.sqrt(m*q+1);
    b1 = (2*np.random.rand(h_size[0], h_size[1])-1) / np.sqrt(m*q+1);   
    w2 = [];
    for i in np.arange(dimout) : 
        w2i = (2*np.random.rand(h_size[0],h_size[1])-1) / np.sqrt(h_size[0]*h_size[1]+1);
        w2.append(w2i);
    b2 = (2*np.random.rand(dimout,1)-1) / np.sqrt(h_size[0]*h_size[1]+1);
    b2 = b2.T[0];
    return w1, b1, w2, b2

#=================================================================
def wsm_convol(xi, xshape, filt, loop, sidevalue=0) :
    ''' Sommme pondérées des Entrées par Produit de convolution du mask
    | En entrée
    | xi     : Un vecteur d'entrée
    | xshape : Les dimensions 2D à appliquer sur le vecteur d'entrée xi
    | filt   : Les poids (partagés) du masque appliqué sur les entrées
    | loop   : Défini les déplacement du mask de haut en bas et de gauche
    |          à droit. Ce paramètre est rendu par la fonction convmask
    | sidevalue : Valeur à attribuer aux points extérieurs à la forme
    |             d'entrée.
    | En Sortie :
    | Ai    : Les produits (sommes pondérées) de convolution
    | sub_x : Les valeurs d'entrée par masque
    '''
    n, p  =  xshape;
    m, q  = np.shape(filt); # (w1)
    
    n_new = n+(2*m)-2;
    p_new = p+(2*q)-2;
    mat   = np.ones((n_new, p_new))*sidevalue; 
    xi    = np.reshape(xi,xshape);
    mat   = wsm_subarr(mat,xi,q-1,m-1);
    
    sub_x  = [];
    
    loopl, loopc = loop;   
    szkl = np.size(loopl);
    szkc = np.size(loopc);
    Ai   = np.zeros((szkl,szkc));
    
    k    = 0;
    for i in loopl :
        sub_xj=[];
        h = 0;
        for j in loopc :
            matij   = mat[i:i+m, j:j+q]; 
            Ai[k,h] = np.sum(matij * filt);
            sub_xj.append(matij);
            h+=1;
        sub_x.append(sub_xj);
        k+=1;
    return Ai, sub_x;


def wsm_out(Fhid,w1,b1,w2,b2,x,xshape,loop,sidevalue=0) : 
    Napp   = np.size(x,0);
    dimout = np.size(b2,0);
    y      = np.zeros(dimout);
    Y      = np.zeros((Napp,dimout));
    for n in np.arange(Napp) :   # pour chaque forme            
        # Forwarding data :
        Ai, sub_x  = wsm_convol(x[n,:], xshape,w1,loop, sidevalue=sidevalue);       
        if Fhid == "lin" : 
            Ai = Ai + b1;
        else : #:=> hidden_function=='tah'   
            Ai = np.tanh(Ai + b1);
        # network output
        for k in np.arange(dimout) :
           y[k] = np.tanh(np.sum(w2[k] * Ai) + b2[k]);
        Y[n,:] = y; # + sto de chaque y calculé
    return Y

#=================================================================
def wsmout(Fhid,w1,b1,w2,b2,x,xshape,skip,sidevalue=0) :
    ''' Y = wsmout(Fhid,w1,b1,w2,b2,x,xshape,skip,sidevalue=0) : Calcul des valeurs
    | de sortie de l'architecture à masque et poids partagés
    | En entrée :
    | Fhid      : Fonction à utiliser pour le calcul de la carte des caractéristiques
    |             (convolution sur les entrées). :
    |             "lin" : fonction linéaire
    |             "tah" ou "tgh" : fonction tangente hyperbolique
    |             Remarque : La couche de sortie est calculée avec la fonction
    |                        tangente hyperbolique
    | w1,b1,w2,b2 : Les poids et les seuils à utiliser pour le calcul de la sortie.    
    | x         : (matrice Nxp) Ensemble des données d'entrées : N est le nombre
    |             de lignes d'exemples, p est la dimension des exemples (c'est à
    |             dire des vecteurs d'entrée en ligne).
    | xshape    : [a, b] Les dimensions 2D à appliquer sur les vecteurs d'entrées
    |             de l'ensemble d'apprentissage.   
    | skip      : [l, c] : Décalage à appliquer pour le masque (mask) d'entrée.
    |              l : décalage horizontal (en nombre de ligne1)
    |              c : décalage vertical (en nombre de colonne)
    | sidevalue : Valeur d'entrée à affecter sur les points de la zone frontalière 
    |             des formes d'entrées
    | En sorties :
    | Y  : Sorties du réseau pour l'ensemble x.
    '''
    if Fhid is not "lin"  and  Fhid is not "tah" and  Fhid is not "tgh" :      
        print("wsmout: ",Fhid,"activation function not defined !");
        sys.exit(0);
    mask = np.shape(w1);
    h_size, loop = wsm_convmask(xshape,mask,skip);
    Y    = wsm_out(Fhid,w1,b1,w2,b2,x,xshape,loop,sidevalue=-1);
    return Y

#=================================================================
def wsmtrain(Xa,Ya,w1,b1,w2,b2,Fhid,xshape,skip,sidevalue=0,   \
             n_epochs=100,lr=0.1,tfp=10,dfreq=10,dprint=1,dcurves=0, dperf=1, \
             Xv=None,Yv=None,pval=0,ntfpamin=100,dval=1) :           
    ''' Apprentissage d'un réseau de neurones à masques et poids partagés pour une
    | architecture particulière : qui ne comprend qu'un niveau de convolution
    | constitué que d'une seule carte de caractéristique.
    | Sans données de validation : w1,b1,w2,b2 = wsmtrain(Xa,Ya,...)
    | Avec données de validation : w1,b1,w2,b2,it_minval = wsmtrain(Xa,Ya,...)
    | 
    | En Entrée :
    | Xa        : (matrice Nxp) Données d'entrées de l'ensemble d'apprentissage.
    |             N est le nombre de lignes d'exemples, p est la dimension des
    |             exemples (c'est à dire des vecteurs d'entrée en ligne).
    | Ya        : (matrice Nxq) Données de sortie de l'ensemble d'apprentissage
    | w1,b1,w2,b2: Poids et seuils initiaux.
    | Fhid      : Fonction à utiliser pour le calcul de la carte des caractéristiques
    |             (convolution sur les entrées). :
    |             "lin" : fonction linéaire
    |             "tah" ou "tgh" : fonction tangente hyperbolique
    |             Remarque : La couche de sortie est calculée avec la fonction
    |                        tangente hyperbolique
    | xshape    : [a, b] Les dimensions 2D à appliquer sur les vecteurs d'entrées
    |             de l'ensemble d'apprentissage.   
    | skip      : [l, c] : Décalage à appliquer pour le masque (mask) d'entrée.
    |              l : décalage horizontal (en nombre de ligne1)
    |              c : décalage vertical (en nombre de colonne)
    | sidevalue : Valeur d'entrée à affecter sur les points de la zone frontalière 
    |             des formes d'entrées
    | n_epochs  : (entier) Nombre maximum d'itérations (defaut: 100)
    | lr        : learning rate (ou pas de gradient) (defaut: 0.1)
    | tfp       : (entier) Top Frequency Processing : Permet de diminuer le nombre 
    |             de certains calcul (eraly stopping) en ne les effectuant que tous
    |             les tfp fois. (defaut: 10)
    | dfreq    : (entier) Fréquence d'affichage. par defaut, qu'un affichage final.    
    | dprint   : (entier) print (ou pas) des valeurs pendant le déroulement de    
    |            l'apprentissage, aussi en fonction des paramètre dperf, dval qui
    |            suivent. Valeur par défaut = 1 : affichage. 
    | dcurves  : Affichage (ou pas) des courbes du déroulement de l'apprentissage
    |            aussi en fonction des paramètre dperf, dval qui suivent.
    |            valeur par défaut = 0 : pas d'affichage.
    |            >= 1 : Affichage
    |               2 : Les erreurs sont affichées en log.
    | dperf    : Affichage (si=1) ou pas (si=0) de valeurs associées aux performances
    |            qui se comprennent ici comme des pourcentages d'erreur.
    | Xv       : (matrice) Données d'entrée de validation. Permet de pratiquer
    |            l'arrêt prématuré.
    | Yv       : (matrice) Sorties désirées associées aux données de validation Xv.
    | pval     : Indique sur quelle valeur doit se faire l'appréciation de l'arrêt
    |            prématuré :
    |            0 : sur l'erreur en validation (c'est le cas pas défaut)
    |            1 : sur la performance en validation 
    | ntfpamin : (entier) Nombre maximum de tfp à faire après le dernier minimum 
    |             constaté en validation (défaut: 100).
    | dval     : Affichge (si=1) ou pas (si=0) des valeurs qui se concerne l'ensemble
    |            de validation
    |
    | En Sortie :
    | Sans données de validation :
    |   w1,b1,w2,b2 : Matrices des poids obtenus à la fin de la procédure.
    | Avec données de validation   :
    |   w1,b1,w2,b2 : Matrices des poids obtenus au minimum sur l'ensemble de
    |                 validation. (Les poids de fin de procédure sont stockés
    |                 dans un fichier: wsmendwei)
    |   it_minval: l'indice de l'itération où le minimum en validation a été trouvé.
    '''
    nbitamin = ntfpamin; #...
    mask = np.shape(w1);
    h_size, loop  = wsm_convmask(xshape,mask,skip); 
    
    # Check and init stuff
    if np.prod(xshape) != np.size(Xa,1) :
        print("wsmtrain: la longueur des entrées n'est pas compatible avec les dimensions données dans xshape");
        sys.exit(0);
    if Fhid is not "lin"  and  Fhid is not "tah" and  Fhid is not "tgh" :      
        print("wsmtrain: ",Fhid,"activation function not defined !");
        sys.exit(0);
    Nout, dimout = np.shape(Ya);
    Napp, papp   = np.shape(Xa);
    if Nout != Napp :
        print("wsmtrain: input and output of learning set must have the same number of item");
        sys.exit(0);

    # Do we use Validation data (for early stopping)
    valid_on = False;
    if Xv is not None and Yv is not None :
        Nxval, pxval = np.shape(Xv);
        if Nxval != np.size(Yv,0) :
            print("wsmtrain: input and output validating set must have the same number of item");
            sys.exit(0);
        if pxval != papp :
            print("wsmtrain: input dim of validating and learning set must be the same");
            sys.exit(0);
        valid_on = True;
        cpit_addmin = 0;  # ComPtage du nbre d'IT apres le Min
        minval   = 1e16;  # Initialisation du minimum en validation
        it_minval = 0;    # Itération du minimum en validation
        WBMV   = [np.copy(w1),np.copy(b1),np.copy(w2),np.copy(b2)];
    if valid_on == False :
        dval = 0;    # Si dval était à 1, on le force à 0.

    # init other stuff
    rw1, cw1 = np.shape(w1); 
    lr2      = lr/max(h_size);
    lr1      = lr/np.sqrt(rw1*cw1);
    y        = np.zeros(dimout);
    #yt      = np.zeros((Napp,dimout));

    #--------------------------------------------------------
    if (dcurves>0) :
        plt.figure();
        xplot = np.array([-1, 0]).astype(int);
        plt.ion();

    #=========================================================
    # boucle principale d'Apprentissage
    continumongars = True;
    epk = 0;
    while (epk < n_epochs) and continumongars :
        epk +=1;
        
        err = [];       
        for n in np.arange(Napp) :   # pour chaque forme d'apprentissage            
            # Forwarding data :
            Ai, sub_x  = wsm_convol(Xa[n,:], xshape,w1,loop, sidevalue=sidevalue);
            
            if Fhid == "lin" : 
                Ai = Ai + b1;
            else : #:=> hidden_function=='tah'   
                Ai = np.tanh(Ai + b1);

            # network output
            for k in np.arange(dimout) :
               y[k] = np.tanh(np.sum(w2[k] * Ai) + b2[k]);
            #yt[n,:] = y; # + sto de chaque y calculé
            
            # Error computation:
            errn = (Ya[n,:] - y);
            err.append(errn);

            dy_db2         = (1-y*y);
            sum_err_dy_dw1 = np.zeros((rw1,cw1));
            sum_err_dy_db1 = np.zeros((h_size[0],h_size[1]));

            for k in np.arange(dimout) :
                dy_dw1_k = np.zeros((rw1,cw1)); 
                dy_db1   = np.zeros((h_size[0],h_size[1]));

                w2_k = w2[k];
                
                if Fhid == "lin" : 
                    for i in np.arange(h_size[0]) :
                        sub_xi = sub_x[i];
                        for j in np.arange(h_size[1]) :
                            #dy_dw1_k   = dy_dw1_k    + np.dot(w2_k[i,j], sub_xi[j]);
                            Z           = np.dot(w2_k[i,j], sub_xi[j]);
                            dy_dw1_k    = dy_dw1_k    + Z;                      
                            dy_db1[i,j] = dy_db1[i,j] + w2_k[i,j];
                else : #:=> hidden_function=='tah'
                    for i in np.arange(h_size[0]) :
                        sub_xi = sub_x[i];
                        for j in np.arange(h_size[1]) :
                            Z = np.dot((1-Ai[i,j]**2),w2_k[i,j]);
                            dy_dw1_k    = dy_dw1_k + np.dot(Z, sub_xi[j])                       
                            dy_db1[i,j] = dy_db1[i,j] + Z;
                
                dy_dw1_k = np.dot(dy_db2[k], dy_dw1_k);
                dy_db1   = np.dot(dy_db2[k], dy_db1);

                sum_err_dy_dw1 = sum_err_dy_dw1 + np.dot(err[n][k], dy_dw1_k);
                sum_err_dy_db1 = sum_err_dy_db1 + np.dot(err[n][k], dy_db1);

                w2[k] = w2_k + np.dot(lr2*dy_db2[k],err[n][k]*Ai);

            b2 = b2 + lr2 * dy_db2 * err[n];
            w1 = w1 + lr1 * sum_err_dy_dw1 / dimout;
            b1 = b1 + lr1 * sum_err_dy_db1 / dimout;

        # fin d'1 passage de toute la base d'apprentissage
        #
        # Top Frequency Processing -----------------------
        if np.remainder(epk,tfp)==0 :  # toutes les tpf passages de la base d'app
            # Je ne stocke rien
            if valid_on :
                Y = wsm_out(Fhid,w1,b1,w2,b2,Xv,xshape,loop,sidevalue=-1);
                if pval : # C'est la perf qui doit servir pour l'early stop.
                    perfval  = tls.classperf(Yv,Y,miss=1);
                    earlyval = perfval;
                else :    # c'est l'erreur quadratique
                    errqval  = np.sum((Yv-Y)**2);
                    earlyval = errqval;
                    
                # Early stopping (stop criterium 1)
                if earlyval < minval : # detection du minimum en validation
                    minval      = np.copy(earlyval); 
                    it_minval   = np.copy(epk);
                    cpit_addmin = 0;
                    # garder les poids du min en validation (MV stand for Min Validation)
                    WBMV   = [np.copy(w1),np.copy(b1),np.copy(w2),np.copy(b2)];                  
                else :
                    cpit_addmin += 1;
                    if cpit_addmin >= nbitamin :
                        print("wsmtrain: Min Val (%f) at it %d then stop %d it after" %(minval,it_minval,nbitamin*tfp));
                        continumongars = False;                
                 
        # Affichage ---------------------------------------
        # Si demandé et selon dfreq (ou fin de boucle pour le dernier affichage)
        if  (dprint>0 or dcurves>0) \
        and (np.remainder(epk,dfreq)==0 or continumongars==0 or epk>=n_epochs) :
            # Il se peut que dfreq ne coincide pas avec tfp, et que rendu ici
            # toutes les valeurs nécéssaires n'aient pas été calculées, le plus
            # simple alors seraient de les calculer ici.
            #
            # Pour l'ens d'App il faut calculer un Y sur l'ensemble de la base
            Y = wsm_out(Fhid,w1,b1,w2,b2,Xa,xshape,loop,sidevalue=sidevalue);
            errqapp = np.sum((Ya-Y)**2);
            errpapp = np.copy(errqapp);    # l'erreur à ploter
            if dcurves>1 :
                errpapp = np.log(errqapp); # l'erreur à ploter en log
            if dperf :
                perfapp = tls.classperf(Ya,Y,miss=1);

            if dval==1 : # display des elt de validation
                Y = wsm_out(Fhid,w1,b1,w2,b2,Xv,xshape,loop,sidevalue=sidevalue);
                errqval = np.sum((Yv-Y)**2);
                errpval = np.copy(errqval);    # l'erreur à ploter
                if dcurves>1 :
                    errpval = np.log(errqval); # l'erreur à ploter en log
                if dperf :
                    perfval = tls.classperf(Yv,Y,miss=1);
  
            # D'abord les prints (si demandés)
            if dprint :
                # L'entête         
                if np.remainder(epk,20*dfreq)==dfreq :
                    print("| epk  |   ErrqApp  ", end='');
                    if dval :
                        print("|   ErrqVal  ", end='');
                    if dperf :
                        print("| PerfApp ", end='');
                        if dval :
                            print("| Perfval ", end='');
                    print("|");
                # Les valeurs
                print("| %4d | %10.6f " %(epk,errqapp), end=''); # l'err tjrs
                if dval :
                    print("| %10.6f " %(errqval), end='');                
                if dperf :         
                    print("|  %.4f " %(perfapp), end='');           
                    if dval :
                        print("|  %.4f " %(perfval), end='');                   
                print("|");

            # Ensuite la figure des courbes (si demandée)
            if dcurves > 0 :
                xplot = xplot+1;  
                if xplot[0] > 0 : # parce que la 1ère fois on a pas encore de valeur précédente valide
                    if dperf :
                        plt.subplot(2,1,2);
                        plt.plot(xplot,[PrevPerfApp,perfapp],'.-');               
                        if dval :
                            plt.plot(xplot,[PrevPerfVal,perfval],'.-');
                        plt.subplot(2,1,1);
                    #
                    plt.plot(xplot,[PrevErrpApp, errpapp],'.-'); #l'err tjrs
                    if dval :
                        plt.plot(xplot,[PrevErrpVal, errpval],'.-');
                    plt.draw();
                # sauvegarde des valeurs de plot précédent
                PrevErrpApp=errpapp; # tjrs
                if dperf :
                    PrevPerfApp=perfapp;
                if dval :
                    PrevErrpVal=errpval;
                    if dperf :
                        PrevPerfVal=perfval; 

    # fin du while : fin de la boucle principale d'Apprentissage
    #=============================================================            
    # Last Action ---------------------------------------
    print("wsmtrain: %d epochs done" %epk);
    #
    if dcurves>0 :
        #plt.xlabel("X %d Epoch" % (tfp));
        plt.xlabel("X %d Epoch" % (dfreq));
        if dperf :
            plt.subplot(2,1,2);           
            plt.ylabel("Perf: % Err. de classif.");
            plt.axis("tight");
            plt.subplot(2,1,1); # pour les erreurs (rem: pas de perf => pas de subplot)
        if dcurves == 1 :
            plt.ylabel("$\sum$ Errors Quadratiques");
        else :
            plt.ylabel("log($\sum$ Errors Quadratiques)");
        plt.axis("tight");
        if dval :
            plt.legend(["Learning","Validating"]);
        else :
            plt.legend(["Learning"]);
        plt.show();
    
    if valid_on :
        np.save("wsmendwei", [w1,b1,w2,b2]); # Sauvegarde des poids en fin d'apprentissage                    
        return WBMV[0], WBMV[1], WBMV[2], WBMV[3], it_minval;
    else :
        return w1, b1, w2, b2;

#=================================================================
def wsmshowccpat(Fhid,w1,b1,w2,b2,x,fr,to,xshape,skip,sidevalue=0) :
    ''' Affichage de l'activation de la carte des caracteristiques
    | wsmshowccpat(Fhid,w1,b1,w2,b2,x,t,fr,to,xshape,skip)
    | En entrée :
    | Fhid      : Fonction à utiliser pour le calcul de la carte des caractéristiques
    |             (convolution sur les entrées). :
    |             "lin" : fonction linéaire
    |             "tah" ou "tgh" : fonction tangente hyperbolique
    |             Remarque : La couche de sortie est calculée avec la fonction
    |                        tangente hyperbolique
    | w1,b1,w2,b2 : Les poids et les seuils à utiliser pour le calcul de la sortie.    
    | x         : (matrice) Ensemble des données d'entrées
    | xshape    : [a, b] Les dimensions 2D à appliquer sur les vecteurs d'entrées
    |             de l'ensemble d'apprentissage.   
    | skip      : [l, c] : Décalage à appliquer pour le masque (mask) d'entrée.
    |              l : décalage horizontal (en nombre de ligne1)
    |              c : décalage vertical (en nombre de colonne)
    | sidevalue : Valeur d'entrée à affecter sur les points de la zone frontalière 
    |             des formes d'entrées
    '''
    dimout = np.size(b2);
    y   = np.zeros(dimout);
    mask = np.shape(w1);
    h_size, loop  = wsm_convmask(xshape,mask,skip);
    nbpat = to-fr+1;
    subpa = np.ceil(np.sqrt(nbpat)).astype(int);
    subpb = np.ceil(nbpat/subpa).astype(int);
    plt.ion();
    plt.figure();
    for n in np.arange(to-fr+1)+fr-1 :
        # Forwarding data :
        Ai, sub_x  = wsm_convol(x[n,:],xshape,w1,loop,sidevalue=sidevalue);
        if Fhid == "lin" : 
            Ai = Ai + b1;
        else : #:=> hidden_function=='tah'   
            Ai = np.tanh(Ai + b1);
        # network output        
        for k in np.arange(dimout) :
           y[k] = np.tanh(np.sum(w2[k] * Ai) + b2[k]);
        #
        isub = n-fr+2;
        plt.subplot(subpb,subpa,isub)
        plt.imshow(-Ai, interpolation='none', cmap=cm.gray);
    plt.show();

#============================================================================
#============================================================================
