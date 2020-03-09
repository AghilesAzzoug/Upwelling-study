def triedacp ():
    """
    Le module triedacp contient les méthodes suivantes dédiées à l'ACP :
        acp         : Calcule les valeurs et vecteurs propres et les nouvelles
                      coordonnées
        phinertie   : Calcule et représente les inerties, et énerties cumulés
                      des axes principaux 
        qltctr2     : Calcule les qualités de représentation et les contributions
                      des individus
        dualacp     : Réalise le dual de l'ACP
        cloud       : Nuage (diagramme de dispersion), utilisable pour les individus
                      comme pour les variables
        corcer      : Cercle des corrélation
        dualcorcer  : Cercle des corrélation pour le dual de l'ACP.
    """
    return None

import numpy as np
import numpy.matlib as nm
import matplotlib.pyplot as plt


def acp (X) :
    ''' VAPU, VEPU, XU = acp(X);
    | Calcul les valeurs propres (VAPU) et des vecteurs propres U (VEPU) de la
    | matrice X'X qui sont retournées dans l'ordre de la plus grande à la plus
    | petite valeur propre. Est également retourné XU qui sont les nouvelles
    | coordonnées des individus (XU=X*U)
    '''   
    # Valeurs et Vecteurs Propres                                    
    XtX = np.dot(X.T,X); 
    VAPU, VEPU = np.linalg.eig(XtX); 
    #
    # Ordonner selon les plus grandes valeurs propres
    idx  = sorted(range(len(VAPU)), key=lambda k: VAPU[k], reverse=True); 
    VAPU = VAPU[idx]; 
    VEPU = VEPU[:,idx];
    #
    # Nouvelles Cordonnées des Individus
    XU = np.dot(X,VEPU);
    return VAPU, VEPU, XU


def phinertie (VAPU, ax=None, ygapbar=0.0, ygapcum=0.0) : # Inertie
    ''' INERTIE, ICUM = phinertie (VAPU)
    | Calcule et retourne l'inertie (INERTIE) et l'inertie cumulée (ICUM)
    | des valeurs propres (VAPU) d'une ACP.
    | Présente les résultats avec une figure de bar sur laquel INERTIE et
    | ICUM sont indiquées. (La récupération de ces valeurs de sortie n'est
    | donc pas nécessairement nécessaire)
    ''' 
    p = np.size(VAPU);
    sumI      = np.sum(VAPU);
    INERTIE   = VAPU/sumI;  
    ICUM      = np.cumsum(INERTIE); 
    #print('\nInertie=', INERTIE, '\nInertie cum.=',ICUM);
    if ax is None :
        fig = plt.figure();
        ax = plt.subplot(111)
    index = np.arange(p);
    ax.bar(index+1,INERTIE,align='center');
    ax.plot(index+1, ICUM, 'r-*');
    for i in range(p) :
        if i != 0 :  # n'affiche pas le premier (il est affiché de suite)
            ax.text(i+1,INERTIE[i]+ygapbar,"{:.4f}".format(INERTIE[i]),horizontalalignment='center');
        ax.text(i+1,ICUM[i]+ygapcum,"{:.4f}".format(ICUM[i]),horizontalalignment='center');
    ax.legend(["Inertie cumulée", "Inertie"], loc=7);
    ax.set_xlabel("Axes principaux");
    ax.set_ylabel("Poucentage d'Inertie des valeurs propres");
    return INERTIE, ICUM;


def qltctr2 (XU, VAPU) :
    ''' QLT, CTR = qltctr2 (XU, VAPU);
    | Dans le cadre d'une acp, dont XU sont les nouvelles coordonnées des
    | individus, et VAPU les valeurs propres, qltctr2 calcule et retourne : 
    | - QLT : Les qualités de réprésentation des individus par les axes
    | - CTR : Les contributions des individus à la formation des axes
    '''
    # Qualité de représentation et Contribution des Individus
    p       = np.size(VAPU); 
    C2      = XU*XU; 
    CTR     = C2 / VAPU; 
    dist    = np.sum(C2,1);          
    Repdist = nm.repmat(dist,p,1);
    QLT     = C2 / Repdist.T; 
    return QLT, CTR;


def dualacp(VAPU, VEPU) :   # Dualité
    ''' XV = dualacp(VAPU, VEPU)
    | Calcule les nouvelles coordonnées des VARIABLES à partir des valeurs et
    | des vecteurs propres de l'ACP. La sortie s'appelle XV mais en réalité il
    | s'agit de X'V. (On a appliqué la "1ere" relation duale : X'vi = ui*sqrt(li))
    '''
    p = np.size(VAPU);
    XV = VEPU*nm.repmat(np.sqrt(VAPU).T,p,1); # Ca s'appelle XV mais c'est X'V
    return XV;   # Nouvelles Cordonnées des variables


#def cloud (NC,pa,po,names=0,shape='o',coul='b',markersize=5, fontsize=11) :
def cloud (NC,pa,po,names=None,shape='o',coul='b',markersize=5, fontsize=11,
           txtcoul='b', holdon=False) :
    '''cloud (NC,pa,po,names,shape,coul,markersize,fontsize)
    | Nuage des Nouvelles Coordonnées (NC), des individus ou variables sur le
    | plan des axes pa-po. Si la variables names (qui est facultative) est passée, elles doit contenir
    | les noms à associer à chacun des points du nuage.
    | NC     : nouvelles coordonnées
    | pa, po : les axes du plan à afficher
    | shape  : la forme des points du nuage
    | coul   : couleur des points du nuage (à choisir parmi les caractère permi
    |          de la fonction plot de matplotlib.
    | markersize : taille des points
    | fontsize   : taille du texte
    '''
    N = np.size(NC,0);
    pa=pa-1;   po=po-1;
    if not holdon :
        plt.figure();
    plt.plot(NC[:,pa], NC[:,po],shape,color=coul,markersize=markersize);
    # Tracer les axes
    xlim = plt.xlim(); plt.xlim(xlim);
    plt.plot(xlim, np.zeros(2));
    ylim = plt.ylim(); plt.ylim(ylim); 
    plt.plot(np.zeros(2),ylim);
    # Ornementation
    plt.xlabel("axe %d" % (pa+1),fontsize=fontsize);
    plt.ylabel("axe %d" % (po+1),fontsize=fontsize);
    #if names != 0 :
    if names is not None :
        for i in range(N) :
            plt.text(NC[i,pa], NC[i,po], names[i],fontsize=fontsize);


def cerclecor () :
    ''' Trace un cercle (de rayon 1 et de centre 0) pour le cercle des corrélations
    '''
    plt.figure();
    # Construire et tracer un cercle de rayon 1 et de centre 0
    t = np.linspace(-np.pi, np.pi, 50); 
    x = np.cos(t);
    y = np.sin(t);
    plt.plot(x,y,'-r');  # trace le cercle 
    plt.axis("equal");
    # Tracer les axes
    xlim = plt.xlim(); plt.xlim(xlim); 
    plt.plot(xlim, np.zeros(2));
    ylim = plt.ylim(); plt.ylim(ylim);  
    plt.plot(np.zeros(2),ylim);

  
def corcer (X,XU,pa,po,varnames,shape='o',coul='b',markersize=8, fontsize=11) :
    '''corcer (X,XU,pa,po,varnames,shape,coul,markersize,fontsize)
    | Dessine le cercle des corrélations (CC)
    | X        : Les données de départ (qui peuvent avoir été transformées ou pas)
    | XU       : Les nouvelles coordonnées des indivisus d'une acp 
    | pa-po    : Le plan d'axe pa-po d'une acp pour lequelle on veut le CC.
    | varnames : Les noms des variables.
    | shape    : La forme des points du nuage
    | coul     : Couleur des points du nuage (à choisir parmi les caractère permi
    |            de la fonction plot de matplotlib.
    | markersize : Taille des points
    | fontsize   : Taille du texte
    '''
    pa=pa-1;   po=po-1;
    p = np.size(XU,1);
    cerclecor();
    # Déterminer les corrélations et les ploter
    XUab = XU[:,[pa,po]];
    W = np.concatenate((X,XUab), axis=1);
    R =  np.corrcoef(W.T);
    a = R[0:p,p]; 
    b = R[0:p,p+1];
    #
    plt.plot(a,b,shape,color=coul,markersize=markersize);    
    for i in range(p) :
        plt.text(a[i],b[i], varnames[i],fontsize=fontsize);
    #
    plt.xlabel("axe %d" % (pa+1),fontsize=fontsize);
    plt.ylabel("axe %d" % (po+1),fontsize=fontsize);
    plt.title("ACP : Cercle des corrélations plan %d-%d" % (pa+1, po+1), fontsize=fontsize);

        
def dualcorcer (XV,pa,po,varnames,shape='o',coul='b',markersize=8, fontsize=11) :
    '''dualcorcer (XV,pa,po,varnames,shape,coul,markersize,fontsize)
    | Cercle des corrélations (qui suit relève du dual si on travail avec des
    | données Normées divisées par sqrt(N-1))
    | XV : Nouvelles coordonnées des variables de l'ACP
    | pa-po    : Le plan d'axe pa-po à afficher
    | varnames : Les noms des variables.
    | shape    : La forme des points pour les variables
    | coul     : Couleur des points (à choisir parmi les caractère permi de la 
    |            fonction plot de matplotlib.
    | markersize : Taille des points
    | fontsize   : Taille du texte    
    '''
    pa=pa-1;   po=po-1;
    p = np.size(XV,0);
    cerclecor();
    #
    plt.plot(XV[:,pa], XV[:,po], shape, color=coul, markersize=markersize);
    for i in range(p) :
        plt.text(XV[i,pa], XV[i,po], varnames[i], fontsize=fontsize);
    #
    plt.xlabel("axe %d" % (pa+1),fontsize=fontsize);
    plt.ylabel("axe %d" % (po+1),fontsize=fontsize);
    plt.title("ACP : Cercle des corrélations plan %d-%d" % (pa+1, po+1),fontsize=fontsize);

