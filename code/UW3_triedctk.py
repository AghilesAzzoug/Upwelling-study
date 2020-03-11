def triedctk():
    """
    Les méthodes ctk : Cartes Topologiques de Kohonen
    findbmus 	: Détermine les référents les plus proches (Best Match Units)
    mbmus 	: Trouve les multiples bmus dans l'ordre des plus proches
    errtopo     : Erreur topologique (cas d'une carte rectangulaire uniquement)
    findhits 	: Calcul les nombres d’éléments captés par les référents
    showmap	: Affiche les variables de la carte
    showmapping	: Montre le déploiement de la carte dans l’espace des données (2 à 2)
    showcarte	: Affichage de la carte selon différents paramètres  
    showbarcell	: Affiche les référents de la cartes en forme de bar
    showprofils	: Montre les référents et/ou leurs données sous forme de courbe
    showrefactiv: Fait apparaitre l’activation des neurones (distance inverse)
                  en fonction des formes présentées
    showrefpat	: Montre les formes (intégrées) captés par les neurones
    cblabelmaj	: Labellisation des référents par vote majoritaire
    reflabfreq  : Tableau de frequences des labels par referent
    cblabvmaj   : Label des référents attribué par vote majoritaire
    cblabfreq   : Label Frequence des 'Labels' des référents    
    label2ind	: Transforme des labels de type string en indice (de type entier)
    mapclassif	: Classifie de données par la carte.
    classifperf : Performance en classification
    """
    return None


import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from triedpy import triedtools as tls


def findbmus(sm, Data=None):
    ''' BMUS = findbmus (sm, Data) :
    | Passer la structure de la carte (sm) d'ou on tirera les référents
    | (sm.codebook) et accessoirement les données pour récupérer en sortie
    | les BMUS (Il devrait bien y avoir une fonction équivalente dans Sompy,
    | mais pour le moment, je ne l'ai pas identifiée)
    | Si Data n'est pas renseignée on prend les données de la structure sm
    | qui sont sensées correspondre aux données d'apprentissage de la carte.
    | Il est suggéré d'appeler une fois cette fonction pour garder les bmus
    | qui sont susceptibles d'être utilisés plusieurs fois par la suite.
    | Incidemment, on remarque que l'on peut obtenir des bmus pour d'autres
    | données (comme celles de test par exemple)
    '''
    # Cette fonction peut etre remplacer par mbmus ; on la garde
    # cependant pour assurer une compatibilité vers le bas.
    if Data == [] or Data is None:  # par defaut on prend celles de sm supposées etre celles d'App.
        Data = sm.data
    nbdata = np.size(Data, 0);
    BMUS = np.ones(nbdata).astype(int) * -1;
    distance = np.zeros(sm.nnodes);
    for i in np.arange(nbdata):
        for j in np.arange(sm.nnodes):
            C = Data[i, :] - sm.codebook[j, :];
            distance[j] = np.dot(C, C);
        Imindist = np.argmin(distance);
        BMUS[i] = Imindist;
    return BMUS


def mbmus(sm, Data=None, narg=1):
    # multiple bmus (cette fonction peut remplacer findbmus)
    # sm   : La structure de la carte topologique
    # Data : Les données pour lesquelles ont veut avoir lss bmus
    #        Par défaut on prend celles associées à la carte qui sont
    #        censé etre des données d'apprentissage
    # narg : Nombre de bmus dans l'ordre des plus proches
    if Data == [] or Data is None:  # par defaut on prend celle de sm supposé etre celles d'App.
        Data = sm.data
    nbdata = np.size(Data, 0);
    MBMUS = np.ones((nbdata, narg)).astype(int) * -1;
    distance = np.zeros(sm.nnodes);
    for i in np.arange(nbdata):
        for j in np.arange(sm.nnodes):
            C = Data[i, :] - sm.codebook[j, :];
            distance[j] = np.dot(C, C);
            # Imindist = np.argmin(distance);
        Imindist = np.argsort(distance);
        MBMUS[i] = Imindist[0:narg];
    return MBMUS


def errtopo(sm, bmus2):  # dans le cas 'RECT' uniquement
    # te : Topographic error, the proportion of all data vectors
    #       for which first AND second BMUs are not adjacent units.
    #              0    1    2    3     4
    #                        |
    #              5    6--- 7 ---8     9
    #                        |
    #             10   11    12  13    14
    #
    # par ex les voisins de 7 :  7-5, 7-1, 7+1, 7+5  (5 etant le nbr de col)
    # c'est donc  :  (7, 7, 7, 7) + (-5, -1, +1, +5) = (2, 6, 8, 12)
    # supposons que le 2ème bmus après 7 soit :
    #   13 -> +1 non adjacent
    #    8 -> +0 non adjacent  
    # bmus2 doit etre un tableau Nx2
    ncol = sm.mapsize[0];
    voisin = np.array([-ncol, -1, +1, +ncol])
    unarray = np.array([1, 1, 1, 1])
    not_adjacent = 0.0;
    for i in np.arange(sm.dlen):
        vecti = unarray * bmus2[i, 0];
        ivois = vecti + voisin
        if bmus2[i, 1] not in ivois:
            not_adjacent = not_adjacent + 1.0;
    et = not_adjacent / sm.dlen
    return et


def findhits(sm, bmus=None):
    ''' HITS = findhits(sm, bmus=None) 
    | Calcul les nombres d’éléments captés par les référents. Ce nombre est
    | appelé hit mais aussi magnitude. Si les bmus ne sont pas passés, on les
    | détermine en considérant les données d'apprentissage.
    '''
    HITS = np.zeros(sm.nnodes).astype(int);
    if bmus == [] or bmus is None:
        bmus = np.array(findbmus(sm));
    for inode in np.arange(sm.nnodes):
        idx = np.where(bmus == inode);
        HITS[inode] = np.size(idx);
    return HITS;


def gatheril(indices, labels):
    sep = ' '
    if len(indices) != len(labels):
        print("indices and Labels must have the same number of items")
        sys.exit(0);
    labs = np.copy(labels).astype(str)
    Flag = (np.ones(max(indices + 1)) * -1).astype(int);
    Uind = (np.ones(len(tls.unique(indices))) * -1).astype(int);
    # Ulab = np.empty(len(tls.unique(indices))).astype(str)
    Ulab = np.empty(len(tls.unique(indices)), dtype='<U64')
    Csep = (np.empty(max(indices + 1))).astype(str);

    k = 0;
    for i in np.arange(len(indices)):
        if Flag[indices[i]] == -1:
            Flag[indices[i]] = k;
            Uind[k] = indices[i];
            Ulab[k] = labs[i];
            Csep[k] = ' ';
            k = k + 1;
        else:
            fii = Flag[indices[i]][0];
            Ulab[fii] = Ulab[fii] + Csep[fii] + labs[i]
            if Csep[fii] == ' ':
                Csep[fii] = '\n';
            else:
                Csep[fii] = ' ';
    return Uind, Ulab


def showmap(sm, sztext=11, coltext='k', colbar=True, cbsztext=8, cmap=cm.jet, interp='none', caxmin=None,
            caxmax=None, axis=None, comp=[], nodes=None, Labels=None, dh=0, dv=0,
            noaxes=True, noticks=True, nolabels=True,
            xticks=None, yticks=None,
            figsize=(12, 16), fignum=None, y=0.98):
    ''' showmap(sm, sztext, colbar, cmap, interp, caxmin,caxmax)
    | Visualisations des variables (componants) de la carte. Il s'agit d'un
    | équivalent moins sophistiqué de som_show (sans la U-Matrix)
    | Les paramètres :
    | sm     : La structure de la carte (C'est le seul paramètre obligatoire)
    | sztext : la taille du text : 11 par défaut
    | colbar : Affichage de la colorbar : True par défaut
    | cmap   : Map de couleur : jet par défaut
    | interp   : Valeur du paramètre d'interpolation pour la fonction imshow de
    |            Matplotlib. Par défaut, ou en présence de None, un lissage des
    |            couleurs est effectué (comme le shading interp de matlab).
    |            Passer la valeur None pour ne pas faire de lissage. 
    | caxmin, et caxmax : permet de définir des bornes min et max communes 
    |         pour les échelles de couleur de toutes les variables.
    | axis   : Permet d'agir sur les axes ('off', equal','tight',...)
    | comp   : Liste d'ndice(s) des composantes (variables) à afficher; par
    |          défaut, elles le seront toutes.
    '''
    nbl, nbc = sm.mapsize;

    if comp is None or comp == []:
        nvar = sm.dim;
        comp = np.arange(nvar);
    else:
        nvar = len(comp)
        comp = np.asarray(comp)
        comp = comp - 1;

    if fignum is not None:
        fig = plt.figure(fignum);
    else:
        fig = plt.figure(figsize=figsize);

    nbsubc = np.ceil(np.sqrt(nvar));
    nbsubl = np.ceil(nvar / nbsubc);
    isub = 0;

    if nodes is not None or Labels is not None:
        if len(nodes) != len(Labels):
            print("triedctk.showmap: nodes and Labels must have the same number of items")
            sys.exit(0);
        Unodes, Ulabels = gatheril(nodes, Labels)

    for i in comp:
        isub += 1;
        ax = plt.subplot(nbsubl, nbsubc, isub);
        Ref2D = sm.codebook[:, i].reshape(nbl, nbc);

        if caxmin is None:
            vmin = np.min(Ref2D);
        else:
            vmin = caxmin;
        if caxmax is None:
            vmax = np.max(Ref2D);
        else:
            vmax = caxmax;

        plt.imshow(Ref2D, interpolation=interp, cmap=cmap, vmin=vmin, vmax=vmax);

        if sm.varname is None or sm.varname is []:
            plt.title("Variable %d" % (isub), fontsize=sztext, y=y);
        else:
            plt.title("%s" % (sm.varname[i]), fontsize=sztext, y=y);
        if colbar == True:
            cbar = plt.colorbar();
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=cbsztext)

        if nodes is not None:
            for n in np.arange(len(Unodes)):
                lig = Unodes[n] // nbc;
                col = Unodes[n] % nbc;
                slen = len(Ulabels[n]);
                dl = 0.015 * slen;  # plus la string est longue plus on décale
                dc = -0.25;  # en ligne et/ou en colone ?
                plt.text(col + dv + dc, lig + dh + dl, Ulabels[n], fontsize=sztext, color=coltext);

        if axis is not None:
            plt.axis(axis)
        #
        if noaxes:
            plt.axis('off')
        elif noticks:
            plt.xticks([]);
            plt.yticks([])
        elif nolabels:
            if xticks is None:
                xticks = plt.xticks()
            if yticks is None:
                yticks = plt.yticks();
            plt.xticks(xticks);
            plt.yticks(yticks);
            ax.tick_params(labelbottom=False, labelleft=False)

    plt.suptitle("Map's components");
    fig.patch.set_facecolor('white');
    return


def showmapping(sm, Data, bmus, seecellid=1, subp=1, override=False,
                refshape='or', refsize=6, grid=1.0,
                gridcolshape='-k', gridligshape='-k'):
    '''showmapping(sm, Data, bmus, subp=1, seecellid=1,override=False,refshape='or',
    | refsize=6,grid=1.0,gridcolshape='-k',gridligshape='-k')
    | showmapping Montre le déploiement de la carte dans l’espace des données
    | par variable 2 à 2.
    | Les paramètres :
    | sm   : La structure de la carte d'où on tirera les référents (sm.codebook)
    | Data : Les données à afficher. Par défaut ce sont les données de la
    |        structure de la carte (sm), qui sont sensées correspondre aux
    |        données d'apprentissage de la carte qui seront utilisé. Incidemment,
    |        on remarque que l'on peut passer d'autres données (comme celles
    |        de test par exemple)
    | bmus : Les bmus des données passées. Si des bmus sont passés, un lien sera
    |        montré entre les référents et les données. Attention de ne pas passer
    |        des bmus qui ne correspondraient pas au données passées, ce qui ne 
    |        peut pas être vérifié.
    | seecellid : Si True, on affiche les numéros des neurones, sinon on ne les 
    |             affiche pas.
    | supb  : Si True les affichages seront fait dans des subplots, sinon il y 
    |         aura une figure pour chaque couple de variable.
    | override : True permet de garder une figure déjà existante ayant déjà 
    |            un nuage de données et de la compléter. Avec False, une nouvelle
    |             figure du nuage des données sera créée.
    | refshape : Permet de choisir la forme et la couleur des référents à
    |            la manière d'un plot (exemple : 'or', '*b')
    | refsize  : Définition de la taille des référents
    | grid     : Epaisseur des traits de la grille de la carte (si 2D)
    |            <=0 : pas de trait de la grille
    |            >0  : epaisseur du trait de la grille de la carte
    | gridcolshape et gridligshape : permettent de choisir la couleur
    |            des traits de grille de la carte respectivement en
    |            colonne et en ligne, à la façon d'un plot (ex: '-k','-g')
                     #       (colonne en vert, ligne en rouge)
    '''
    codebook = sm.codebook;
    msz0, msz1 = sm.mapsize;

    if Data == [] or Data is None:  # Par defaut on prend les données de sm qui
        Data = sm.data;  # sont en principe celles d'apprentissage
    else:  # on s'assure que les données passée on la meme dim que les codebook
        if Data.ndim != codebook.ndim:
            print("showmapping : Data must have the same dim as codebooks");
            sys.exit(0);

    if bmus != [] and bmus is not None:
        if np.size(bmus) != np.size(Data, 0):
            print("showmapping : Data and bmus must have the same number of element");
            sys.exit(0);

    if subp == True:
        plt.figure();
        nbsub = sm.dim * (sm.dim - 1) / 2;  # Cette façon de faire
        nbc = np.ceil(np.sqrt(nbsub));  # vaut pour le cas des
        nbl = np.ceil(nbsub / nbc);  # variables 2 à 2
        isub = 0;
    Ref2D = np.array((sm.nnodes, 2))
    for i in np.arange(sm.dim - 1):
        for j in (np.arange(sm.dim - i - 1) + i + 1):
            if subp == 1:
                isub = isub + 1;
                plt.subplot(nbl, nbc, isub);
            elif override == False:  # cette façon
                plt.figure();  # de faire n'est
            if override == False:  # pas claire
                plt.plot(Data[:, i], Data[:, j], '*y');
            #
            Ref2D = codebook[:, [i, j]];
            #
            # Positionnement des Réferents
            for k in np.arange(sm.nnodes):
                plt.plot(Ref2D[k, 0], Ref2D[k, 1], refshape, markersize=refsize);  #
                if seecellid == True:  # Ajouter un indice des neurones (mais est-ce le bon ?)
                    plt.text(Ref2D[k, 0], Ref2D[k, 1], k + 1, fontsize=13);
                if bmus is not [] and bmus is not None:  # Si Les BMUS sont passés ca signifie
                    # qu'on veut faire apparaitre les liens entre le reférent et les données
                    idx = np.where(bmus == k);
                    if np.size(idx) > 0:
                        for lnk in idx[0]:
                            plt.plot([Ref2D[k, 0], Data[lnk, i]], \
                                     [Ref2D[k, 1], Data[lnk, j]], '-b', linewidth=1);
                            #
            if grid > 0:
                # Grille : Trait selon les colonnes de la carte
                for c in np.arange(msz1):
                    for l in np.arange(msz0 - 1):
                        X = msz1 * l + c
                        Y = X + msz1;
                        plt.plot([Ref2D[X, 0], Ref2D[Y, 0]], \
                                 [Ref2D[X, 1], Ref2D[Y, 1]], gridcolshape, linewidth=grid);
                #
                # Grille : Trait selon les lignes de la carte
                for l in np.arange(msz0):  # Trait en ligne
                    k = l * msz1;
                    for c in np.arange(msz1 - 1):
                        plt.plot([Ref2D[k + c, 0], Ref2D[k + c + 1, 0]], \
                                 [Ref2D[k + c, 1], Ref2D[k + c + 1, 1]], gridligshape, linewidth=grid);


def showcarte(sm, figlarg=8, fighaut=6, hits=None, shape='o', shapescale=300, \
              colcell='b', cmap=cm.jet, text=None, dv=0.0, dh=0.0, coltext='k', \
              sztext=11, showcellid=None):
    ''' showcarte (sm, figlarg, fighaut, hits, shape, shapescale, colcell, cmap,
    |                  text,dv, dh, coltext, sztext)
    | Affichage de la carte selon les paramètres ci-dessous
    | Les paramètres :
    | sm      : La structure de la carte
    | figlarg : Définie la largeur de la figure (valeur standard : 8 par défaut)
    | fighaut : Définie la hauteur de la figure (valeur standard : 6 par défaut)
    | hits    : Prise en compte ou pas des hits c'est à dire du nombre d'éléments
    |           captés par chaque neurone (magnitude). 3 possibilités :
    |           - 'Yes' : on déterminera les hits correspondant aux données
    |                     d'apprentissage de la carte (avec la fonction findhits).
    |           - un vecteur, de même taille que le nombre de neurones de
    |             la carte. Ca peut être des hits comme ceux produit par la fonction
    |             findhits, ou ca peut être tout autre chose jugé opportun.
    |           Dans ces 2 premiers cas, les neurones seront dimensionnés
    |           proportionnellement aux valeurs du vecteur passé.
    |           - Sinon (3ème cas) Les neurones seront tous de la même taille
    | shape   : caractère à utiliser pour dessiner les formes neurones. les valeurs
    |           possibles sont celles de la fonction plot de matplotlib ('v','p',
    |          '*','d', ...) 'o' est la valeur par défaut.
    | shapescale : Echelle de taille pour de la forme des neurones
    | colcell : Couleur à utiliser pour les cellules :
    |           - soit un des caractères possible de la fonction plot : 'w','k','r',
    |            'y','g','m','c' et 'b' qui est la valeur par défaut.
    |           - soit un tableau d'entiers (de la taille du nombre de neurones), sur
    |             lequel sera étalonné la map de couleurs.
    | cmap    : Map de couleur : jet par défaut. Inutile si on a choisi, même 
    |           implicitement, (c'est à dire par défaut), un caractère de couleur
    |           pour le paramètre colcell précédent.
    | text    : Liste (tableau?) de chaînes de caractères à afficher sur les neurones
    |           Il doit y en avoir autant que de neurones.
    | dv, dh  : Décalage vertical(dv) et horizontal (dh) pour ajuster le text
    |           au mieux sur les neurones.
    | coltext : Couleur à appliquer au paramètre text précédent à choisir parmi les
    |           caractères possibles du plot de matplotlib : 'w','b','r','y','g',
    |           'm','c' et 'k' qui est la valeur par défaut.
    | sztext  : Taille du texte (fontsize), 11 par défaut.
    | showcellid : si True (ou 1), le n° de la cellule est affiché.
    '''
    Data = sm.data;
    Mmsz = np.max(sm.mapsize)
    mmsz = np.min(sm.mapsize)
    msz0, msz1 = sm.mapsize;

    fig = plt.figure(figsize=(figlarg, fighaut));  # taille std par defaut (8, 6)
    # shapescale = 300; # 351 : Echelle qui semble bien aller avec
    #                  # la taille (8,6) de la figure par defaut
    delta = 1 / Mmsz;
    midelta = delta / 2;
    left = (1 - delta * mmsz) / 2;

    if hits == 'Yes':
        hits = findhits(sm);

    if hits != [] and hits is not None:
        if np.size(hits, 0) != sm.nnodes:
            print("showcarte: le vecteur hits doit avoir autant d'élément qu'il y a de neurones");
            sys.exit(0);
        hitmax = np.max(hits);
        hitdscale = delta * shapescale / hitmax;
        WithHits = True;
    else:
        mksize = delta * shapescale
        WithHits = False;

    WithText = False;
    if text != [] and text is not None:
        if np.size(text, 0) != sm.nnodes:
            print("showcarte: Si on met du texte il doit y avoir une string pour chaque neurone");
            sys.exit(0);
        WithText = True;

    if type(colcell) != str:
        if np.size(colcell, 0) != sm.nnodes:
            print("showcarte: size of color array (colcell) must be equal the number of cells");
            sys.exit(0);
        colcellmax = max(colcell) + 1;
        Tcol = cmap(np.arange(1, 256, round(256 / colcellmax).astype(int)));
        Tcol[:, 3] = 0.5;
        print(Tcol);

    inode = 0;
    for i in np.fliplr([np.arange(msz0)])[0]:  # les lignes en y
        for j in np.arange(msz1):  # les colonnes en x
            if WithHits:
                mksize = hits[inode] * hitdscale;
            if msz1 > msz0:
                jx = j * delta + midelta;
                iy = i * delta + midelta + left;
                if type(colcell) != str:
                    plt.plot(jx, iy, shape, markersize=mksize, color=Tcol[colcell[inode]]);
                else:
                    plt.plot(jx, iy, shape, markersize=mksize, color=colcell);
            else:
                jx = j * delta + midelta + left;
                iy = i * delta + midelta;
                if type(colcell) != str:
                    plt.plot(jx, iy, shape, markersize=mksize, color=Tcol[colcell[inode]]);
                else:
                    plt.plot(jx, iy, shape, markersize=mksize, color=colcell);

            if WithHits and showcellid:
                plt.text(jx - 0.03, iy + midelta - 0.03, "%d|%d" % (inode + 1, hits[inode]), fontsize=sztext);
            elif showcellid:
                plt.text(jx, iy + midelta, "%d" % (inode + 1), fontsize=sztext);
            elif WithHits:
                plt.text(jx, iy + midelta, "%d" % (hits[inode]), fontsize=sztext);
            if WithText:
                plt.text(jx + dv, iy + dh, text[inode], fontsize=sztext, color=coltext, );
            inode += 1;

    plt.suptitle("La Carte", fontsize=sztext);
    plt.axis([0, 1, 0, 1]);
    plt.axes().set_aspect('equal')
    plt.axis('off');


def showbarcell(sm, norm='brute', a=0, b=1, scale=0, cmap=cm.rainbow, sztext=11):
    ''' showbarcell (sm,norm,a,b,scale,cmap,sztext)
    | Représente chaque référent sous forme de bar dans un subplot.
    | Les paramètres :
    | sm   : La structure de la carte
    | norm : (string) Le codage des référents, il y a 3 possibilités :
    |        - 'brute' : Les référents ne sont pas codés (c'est la valeur par défaut)
    |        - 'varia' : Les référents sont centrés et réduits.
    |        - 'range' : Les référents sont normé dans un intervalle [a, b], dont
    |                    les valeurs sont indiquées par les paramètres qui suivent.
    | a et b : Borne inf et sup de l'intervalle à appliquer dans le cas d'un codage
    |          'range' du paramètre norm précédent (sinon, ces paramètres ne sont
    |          pas utiles. Valeurs par défaut : a=0 et b=1
    | scale     : Echelle pour les axes des subplots
    |             1 : échelle indépendante ajustée : axis('tight')  
    |             2 : échelle commune pour tous les subplots comprise entre le min et
    |                 le max de tous les référents.
    |             sinon, l'échelle n'est pas définie (valeur par défaut). Il se
    |             pourrait bien que se soit axis('tight') qui est appliqué dans ce
    |             cas ?...)
    | cmap   : Map de couleur des bars : rainbow par défaut. 
    | sztext : Taille du texte (fontsize), 11 par défaut.
    '''
    nbl, nbc = sm.mapsize
    D = sm.codebook;

    Tcol = cmap(np.arange(1, 256, round(256 / sm.dim)));
    index = np.arange(sm.dim);

    if norm == 'varia':
        D = tls.centred(D);
    elif norm == 'range':
        D = tls.normrange(D, a, b);
    mi = np.min(D);
    ma = np.max(D);

    fig = plt.figure();
    isub = 0;
    for i in np.arange(sm.nnodes):
        plt.subplot(nbl, nbc, isub + 1);
        bhdl = plt.bar(index + .6, D[isub, :]);
        for j in np.arange(sm.dim):
            bhdl[j].set_color(Tcol[j, :]);
        if scale == 1:
            plt.axis("tight");
        elif scale == 2:
            plt.axis([0.5, sm.dim + 0.5, mi, ma]);
        isub += 1;
        plt.title(isub, fontsize=sztext);
    plt.suptitle("cells codebook bar");
    fig.patch.set_facecolor('white');
    return


def showprofils(sm, visu=1, Data=None, bmus=None, scale=None, \
                Clevel=None, Gscale=0.25, showcellid=None, ColorClass=None,
                sztext=11, axsztext=8, markrsz=6, marker='*', pltcolor='r', xticks=None,
                ticklabels=None,
                axline=False,
                figsize=(12, 16), fignum=None, y=0.98, verbose=False):
    ''' showprofils (sm, visu, Data, bmus ,scale, Clevel, Gscale)
    | Pour chaque neurone, on représente, dans un subplot, le référent et/ou des
    | données qu'ils a captées sous forme de courbe.
    | Les paramètres :
    | sm     : La structure de la carte
    | visu   : Indicateur d'affichage :
    |          1: que les référents;   2: que les données;   3: ref+données 
    | Data   : Les données à affichées. N'est utile que si visu = 2 ou 3. Par défaut
    |          ce sont les données d'apprentissage de la carte qui seront utilisées.
    |          Incidemment, on remarque que l'on peut passer d'autres données
    |          (comme celles de test par exemple)
    | bmus   : Si les données doivent être affichées (visu=2 ou 3) et que des bmus
    |          sont passés, ce doit être ceux associées aux données passées (Data).
    |          les résultats sont imprévisibles sinon. Par défaut on associera les
    |          référents aux données d'apprentissage de la carte dans la structure
    |          sm.
    | sztext   : Taille du texte des titres (fontsize), 11 par défaut.
    | axsztext : Taille du texte des axes(fontsize), 8 par défaut.
    | scale  : Echelle pour les axes des subplots
    |          1 : échelle indépendante ajustée (axis('tight'))  
    |          2 : échelle commune pour tous les subplots comprise entre le min et
    |              le max de toutes les valeurs à afficher.
    |          sinon : echelle non specifiée. (ce peut être axis('tight')?)
    | Clevel : Vecteur de la taille du nombre de neurones : Indices pour
    |          différencier des groupes de neurones avec des niveaux de gris
    |          différents (ca peut être des indices de classe par exemple). Par
    |          defaut, le fond sera blanc.
    | Gscale : (réel entre 0 et 1 : Permet de jouer un peu sur les nuances des
    |          niveaux de gris (=0.25 par defaut) lorsque Clevel est utilisé.
    | showcellid : si True (ou 1), le n° de la cellule est affiché. ###<<<
    '''
    #
    nbl, nbc = sm.mapsize;
    if visu == 2 or visu == 3:  # Si on veut des données
        if Data == [] or Data is None:  # Par defaut on prend les données de sm qui
            Data = sm.data;  # sont en principe celles d'apprentissage
        else:  # on s'assure que les données passée on la meme dim que les codebook
            if Data.ndim != sm.codebook.ndim:
                print("showprofils : Data must have the same dim as codebooks");
                sys.exit(0);

        if bmus == [] or bmus is None:
            bmus = findbmus(sm, Data);
            # Ici on peut vérifier que bmus et data ont la même taille
        if np.size(bmus) != np.size(Data, 0):
            print("showprofils : Data and bmus must be the same size");
            sys.exit(0);
        minX = np.min(np.concatenate((Data, sm.codebook), 0));
        maxX = np.max(np.concatenate((Data, sm.codebook), 0));
    else:  # Si on ne veut que les référents
        minX = np.min(sm.codebook);
        maxX = np.max(sm.codebook);
    #        
    # Color stuff pour éventuellement différencier les référents selon un critère
    if ColorClass is None:
        if Clevel == [] or Clevel is None:
            ColorRef = np.ones((sm.nnodes, 3));  # par défaut du blanc pour tous les fonds de cellule
        else:
            if np.size(Clevel) != sm.nnodes:
                print("showprofis: size of color indices (Clevel) must be equal the number of cells");
                sys.exit(0);
            # Prévoir un étalement des niveaux de gris entre Gscale(>0) et
            b = 1;
            if Gscale < 0 or Gscale > 1:
                print("showprofils: Gscale must be between 0 and 1 : try again");
                sys.exit(-1);
            minx = min(Clevel);
            maxx = max(Clevel);
            dx = maxx - minx;
            dr = b - Gscale;
            colorClass = Gscale + dr * (Clevel - minx) / dx;  # dans [Gscale b]
            ColorRef = np.array([colorClass, colorClass, colorClass]).T;

    if visu < 1 or visu > 3:
        print("showprofils : bad visu value -> turn to 1 (referents only)");
        visu = 1;
    #
    if fignum is not None:
        fig = plt.figure(fignum);
    else:
        fig = plt.figure(figsize=figsize);
        fignum = fig.number  # numero de figure en cours ...
    #
    # Creationdes SUBPLOTS ....
    if visu == 1 or visu == 2 or visu == 3:
        # Utilisation de 'subplots' en une fois a la place de 'subplot' a chaque iteration.
        # Cela evite de recreer un axe (ax) sur la place d'un ancien lors de la deuxieme boucle
        # et a eviter le warning :
        #   /Applications/Anaconda/anaconda3-5.2.0/envs/python3/lib/python3.6/site-packages/
        #    matplotlib/cbook/deprecation.py:107: MatplotlibDeprecationWarning: Adding an axes
        #       using the same arguments as a previous axes currently reuses the earlier instance.
        #       In a future version, a new instance will always be created and returned.  Meanwhile,
        #       this warning can be suppressed, and the future behavior ensured, by passing a unique
        #       label to each axes instance.
        fig, axarr = plt.subplots(nrows=nbl, ncols=nbc, num=fignum, facecolor='w')
    #
    if visu == 2 or visu == 3:  # Les données
        inode = 0;
        for l in np.arange(nbl):  # en supposant les référents
            for c in np.arange(nbc):  # numerotés de gauche à droite
                ax = axarr[l, c]
                idx = np.where(bmus == inode);
                if np.size(idx) > 0:
                    ax.plot(Data[idx[0], :].T, '-b', linewidth=2);
                #
                inode += 1;
                # To have Neurone indice number (if required)
                if showcellid:
                    ax.title("Cell N° %d" % (inode), fontsize=sztext);
                    #
    if visu == 1 or visu == 3:  # Les référents
        inode = 0;
        for l in np.arange(nbl):  # en supposant qu'ils sont
            for c in np.arange(nbc):  # numerotés de gauche à droite
                ax = axarr[l, c]
                #
                ax.plot(sm.codebook[inode, :], '-', color=pltcolor, linewidth=0.5);
                ax.plot(sm.codebook[inode, :], marker=marker, markersize=markrsz, color=pltcolor);
                #
                if axline:
                    ax.axhline(color='k', linewidth=0.5)
                #
                if scale == 1:
                    ax.axis("tight");
                elif scale == 2:
                    ax.axis([0, sm.dim - 1, minX, maxX]);
                if 1:  # Couleur de fond qu'il faut faire correspondre à la celle de la classe
                    from matplotlib import __version__ as plt_version
                    if plt_version < '2.0.':
                        ax.set_axis_bgcolor(ColorClass[Clevel[inode]])
                    else:
                        ax.set_facecolor(ColorClass[Clevel[inode]])
                        # axe values only in borders of subplots
                ax.tick_params(labelsize=axsztext)
                if l < (nbl - 1):
                    ax.tick_params(labelbottom=False)
                elif ticklabels is not None:
                    if xticks is None:
                        xticks = ax.xticks()[0];
                    if verbose:
                        print(xticks)
                        print(ticklabels)
                    localtcks = []
                    locallbls = []
                    for ii in np.arange(len(ticklabels)):
                        if verbose:
                            print(ii, np.where(xticks == float(ii))[0])
                        if len(np.where(xticks == float(ii))[0]) > 0:
                            if ii < len(ticklabels):
                                localtcks.append(ii)
                                locallbls.append(ticklabels[ii])
                            else:
                                localtcks.append(ii)
                                locallbls.append('')
                        else:
                            localtcks.append(ii)
                            locallbls.append('')
                    ax.set_xticks(localtcks)
                    ax.set_xticklabels(locallbls, rotation=60)
                else:
                    if xticks is not None:
                        ax.set_xticks(xticks)
                #
                if c > 0:
                    ax.tick_params(labelleft=False)
                inode += 1;

    fig.patch.set_facecolor('white');
    return


def showrefactiv(sm, Patterns, pattext=None, sztext=11, cmap=cm.jet, interp='none'):
    ''' showrefactiv(sm, Patterns, pattext, sztext, cmap, interp)
    | Cette méthode montre comment s'active les neurones de la carte lorsque
    | une forme lui est présentée. En fait on calcul la distance (euclidienne)
    | entre la forme et les référents. L'activation des neurones sera la valeur
    | opposée à cette distance. Autrement dit, plus la distance est petite, plus
    | la valeur du neurone en réaction sera élévée.
    | Les paramètres :
    | sm       : La structure de la carte
    | Patterns : Les formes pour lesquelles on veut voir l'activation des neurones
    |            de la carte. Il y aura un subplot pour chaque forme. Il serait
    |            donc raisonnable de ne visualiser que quelques formes à la fois.
    | pattext  : Un tableau de string à afficher pour chaque forme.
    | sztext   : Taille du texte (fontsize), 11 par défaut.
    | cmap     : Map de couleur : jet par défaut.
    | interp   : Valeur du paramètre d'interpolation pour la fonction imshow de
    |            Matplotlib. Par défaut, ou en présence de None, un lissage des
    |            couleurs est effectué (comme le shading interp de matlab).
    |            Passer la valeur None pour ne pas faire de lissage. 
    '''
    if Patterns.ndim != sm.codebook.ndim:
        print("showrefactiv : Patterns must have the same dim as codebooks");
        sys.exit(0);
    nb = np.size(Patterns, 0)
    Dist = tls.vdist(sm.codebook.T, Patterns.T);

    fig = plt.figure();
    nbsubc = np.ceil(np.sqrt(nb));
    nbsubl = np.ceil(nb / nbsubc);
    isub = 0;
    for i in np.arange(nbsubl):
        for j in np.arange(nbsubc):
            if isub < nb:
                Activi = np.reshape(Dist[isub, :], sm.mapsize)
                isub = isub + 1;
                plt.subplot(nbsubl, nbsubc, isub);
                plt.imshow(Activi * -1, cmap=cmap, interpolation=interp);
                if pattext != [] and pattext is not None:
                    plt.title(pattext[isub - 1], fontsize=sztext);
    plt.suptitle("Cell's activation from pattern(s) input (opposite distance)");
    fig.patch.set_facecolor('white');
    return


def showrefpat(sm, x, x0, x1, bmus=None, Data=None, hits=None,
               axis=None, box=None, grid=None, ticks=None,
               sztext=11, coltext='k', titlepos=None, colfig='white'):
    '''showrefpat(sm,x,x0,x1, bmus, Data, hits, sztext)
    | Visualisation de formes qui peuvent être les données d'apprentissage elles-
    | même ou bien des données qui leurs sont associées. Ces forme sont montrées
    | pour chaque référent en intégrant toutes les formes que le référents a
    | captées.
    | paramètres :
    | sm     : La structure de la carte
    | x      : Les patterns à visualiser
    | x0, x1 : le redimensionnement à opérer sur les patterns pour les visualiser
    |          en 2D (reshape(pattern,[x0, x1]).
    | bmus   : Les bmus à associer aux parttern. A défaut, on déterminera les bmus
    |          des données passées, à défaut de ces dernières on prendra les bmus
    |          des données d'apprentissage de la carte. A la fin il doit y avoir
    |          autant de bmus que de patterns à visualiser.
    | Data   : Une matrice de données d'entrée de la carte (d'apprentissage, de test
    |          ou autre), associées aux patterns à visualiser, et sur lesquels on
    |          s'appuyer pour déterminer les bmus à associer à x si ils (les bmus)
    |          ne sont pas passés.
    | hits   : Si ils sont connus, on peut passer les hits, sinon la méthode les
    |          déterminera elle-même. Il ne sont utilisés que pour indiquer conbien
    |          de patterns on participés à la sommation pour chaque référent.
    | axis   : Permet d'agir sur les axes ('off', equal','tight',...)
    | box, grid : 'on' ou 'off' : permet d'agir sur la présentation de la figure
    | ticks  : 'off' : permet de supprimer les ticks des axes.
    |     Remarque: il peut y avoir des contrariétés entre axis,box,grid,ticks
    |     par exemple, axis('off') ne permet pas d'avoir box('on')
    | sztext : Taille du texte (fontsize), 11 par défaut.
    | coltext: couleur du texte (à la manière de plot)
    | titlepos : Couple de scalaire (a,b) de positionnement du title
    | colfig : Couleur de fond de la figure.
    |          Soit une srting (exemple : 'gray', 'r', ...)
    |          ou un vecteur RGB (exemple : [0.2, 0.6, 0.8])
    '''
    if bmus == [] or bmus is None:
        if Data == [] or Data is None:
            bmus = findbmus(sm);  # on prend les bmus correspond aux données d'app
        else:
            bmus = findbmus(sm, Data);
    # on peut vérifier que x et bmus sont de même taille
    nlx, ncx = np.shape(x);
    if nlx != np.size(bmus):
        print("showrefpat: pattern x and bmus must have the same number of element")
        sys.exit(0);

    if x0 * x1 != ncx:
        print("showrefpat error : x0*x1 must be equal the number of column of x");
        sys.exit(0);

    if titlepos is not None:
        if len(titlepos) != 2:
            print("showrefpat error : titlepos must be a sclare couple");
            sys.exit(0);
        xtitle, ytitle = titlepos;

    if hits == [] or hits is None:
        hits = findhits(sm, bmus)

    nbl, nbc = sm.mapsize;
    SPR = np.zeros((ncx, sm.nnodes));  # Initialisation (a 0) de la variable de sommation
    for i in np.arange(nlx):
        SPR[:, bmus[i]] = SPR[:, bmus[i]] + x[i, :];

    fig = plt.figure();
    SPR = -SPR;  # pour avoir de pattern en noir sur fond blanc
    for k in np.arange(sm.nnodes):
        image = np.reshape(SPR[:, k], [x0, x1])
        plt.subplot(nbl, nbc, k + 1);
        plt.imshow(image, cmap=cm.gray);
        if titlepos is None:
            plt.title("%d-%d" % (k + 1, hits[k]), fontsize=sztext, color=coltext);
        else:
            plt.title("%d-%d" % (k + 1, hits[k]), fontsize=sztext, color=coltext, position=(xtitle, ytitle));

        if axis is not None:
            plt.axis(axis);
        if box is not None:
            plt.box(box);
        if grid is not None:
            plt.grid(grid);
        if ticks == 'off':
            plt.xticks([]);
            plt.yticks([]);

    plt.suptitle("Units summation of patterns associeted to the captured data")
    fig.patch.set_facecolor(colfig);  # ('white');
    return


# **********************************************************************
def cblabelmaj(sm, Dlabel, bmus=None):
    # Cette fonction, dédiée uniquement aux données d'apprentissage pourrait
    # etre dépréciée pour etre remplacée par reflabfreq et cblabvmaj. Elle est
    # néanmoins concervée pour assurer une compatibilité vers le bas
    ''' CBLABEL = cblabelmaj(sm,Dlabel,bmus)
    | Labelisation des référents (CodeBooks) par vote majoritaire uniquement.
    | En entrée :
    | sm     : La structure de la carte
    | Dlabel : Les labels des données d'apprentissage (sous forme de tableau de
    |          chaine de caractère)
    | bmus   : Les bmus des données d'apprentissage
    | En sortie :
    | CBLABEL : contient les labels des référents attribués par vote majoritaire.
    |           Remarque : pour ne pas interférer avec le code Sompy, ces labels
    |           ne sont pas intégrés dans la structure de la carte sm. Ils sont
    |           sous forme de liste de string.
    '''
    if np.size(Dlabel) != sm.dlen:
        print("cblabelmaj : Il doit y avoir autant de labels que de données d'apprentissage");
        sys.exit(0);

    if bmus == [] or bmus is None:
        bmus = findbmus(sm);
    else:  # on verifie qu'il y a autant de bmus que de données d'apprentissage
        if np.size(bmus, 0) != sm.dlen:
            print("cblabelmaj : Il doit y avoir autant de bmuss que de données d'apprentissage");
            sys.exit(0);

    CBLABEL = [];  # Init les labels des référents par vote majoritaire
    for inode in np.arange(sm.nnodes):
        idx = np.where(bmus == inode)  # indice des données de inode
        Ilabel = Dlabel[idx];  # labels de ces données
        # Maintenant faut trouver la classe la plus fréquente dans Ilabel
        szIlabel = np.size(Ilabel);
        if szIlabel > 0:
            Tlab = [];  # Stock les labels rencontrés dans Ilabel
            Clab = [];  # Comptage des labels
            for i in np.arange(szIlabel):
                found = False;
                for j in np.arange(np.size(Tlab)):
                    if Ilabel[i] == Tlab[j]:
                        Clab[j] += 1;
                        found = True;
                        break;
                if found == False:  # C'est un nouveau label pour le Tableau
                    Tlab.append(Ilabel[i]);
                    Clab.append(1);
            imax = np.argmax(Clab)
            CBLABEL.append(Tlab[imax]);
        else:
            CBLABEL.append("")  # Ce réferent n'a rien capté
    return CBLABEL


def reflabfreq(sm, Data, Dlabel, bmus=None):
    # Tableau de frequences des labels par referent
    ''' Tfreq, Ulab = reflabfreq(sm,Data,Dlabel,bmus=None)
    | Tableau de frequences des labels par referent selon les Data qu'ils
    | ont captées.
    | En entrée :
    | sm     : La structure de la carte
    | Data   : Les données à considérer (ce peut etre des données d'apprentssage
    |          de test, ou autres)
    | Dlabel : Les labels des données (sous forme de tableau de chaine de caractères)
    |          Il doit y en avoir autant que de donnée
    | bmus   : Les bmus des données. Si ils ne sont pas passés, ils seront déterminés.
    | En sortie :
    | Tfreq  : Le tableau par référents (en ligne) qui contient les nombres de
    |          données captées par label (en colonne selon Ulab) 
    | Ulab   : Labels (unique) à laquelle correspond chaque colonne
    '''
    if len(Dlabel) != len(Data):
        print("\nreflabelfreq : Il doit y avoir autant de labels que de données");
        sys.exit(0);

    if bmus == [] or bmus is None:
        bmus = findbmus(sm, Data);
    else:  # On verifie qu'il y a autant de bmus que de donnée (ce test concerne
        if len(bmus, 0) != len(Xtest):  # surtout le cas où les bmus sont passés)
            print("\nreflabelfreq : Il doit y avoir autant de bmuss que de données");
            sys.exit(0);

    Ulab = np.unique(Dlabel);  # ['L', 'M', 'T']
    nbUlab = len(Ulab);  # 3

    Tfreq = np.zeros((sm.nnodes, nbUlab)).astype(int);
    for inode in np.arange(sm.nnodes):
        idx = np.where(bmus == inode)  # indices des données de inode
        # pour inode=17 : [286, 333, 394, 397, 461]
        Ilabel = Dlabel[idx];  # labels de ces données : ['M', 'L', 'L', 'L', 'L']
        for i in np.arange(nbUlab):
            ilabi = np.where(Ilabel == Ulab[i])[0];
            Tfreq[inode, i] = len(ilabi);
    return Tfreq, Ulab


def cblabvmaj(Tfreq, Ulab):
    ''' CBLABELS = cblabvmaj(Tfreq,Ulab)
    | Label des référents (cb pour Codebook) attribué par vote majoritaire
    | En entrées : Tfreq,Ulab : les sorties de la methode reflabfreq
    '''
    nnodes = len(Tfreq);
    CBLABELS = [];
    for i in np.arange(nnodes):
        imax = np.argmax(Tfreq[i]);
        if Tfreq[i, imax] == 0:  # le neurone i n'a rien capté"
            CBLABELS.append('');
        else:
            CBLABELS.append(Ulab[imax]);
    return CBLABELS


def cblabfreq(Tfreq, Ulab, csep=' '):
    ''' CBLABELS = cblabfreq(Tfreq,Ulab,csep=' ')
    | Frequence des 'Labels' des référents (cb pour Codebook)
    | En entrées :
    | - Tfreq,Ulab :les sorties de la methode reflabfreq
    | - Le caractère de séparation de chaque frequence de label
    '''
    nnodes = len(Tfreq);
    nlabels = len(Ulab)
    CBLABELS = [];
    for i in np.arange(nnodes):
        strlabi = ''
        for j in np.arange(nlabels):
            if Tfreq[i, j] != 0:
                strlabij = "%s(%d)%c" % (Ulab[j], Tfreq[i, j], csep)
                strlabi = strlabi + strlabij
        CBLABELS.append(strlabi)
    return CBLABELS


def label2ind(label, labelnames):
    ''' INDLABEL = label2ind(label, labelnames)
    | Passage de labels sous forme de string à des indices de label (à partir
    | de 1). L'indice est attribué selon l'ordre des noms de labels indiqués
    | dans labelnames. Exemple :
    |     Si label         = array(['AA','CC','AA','BB','CC','BB','AA']) et que
    |     labelnames       = ['BB','AA','CC']
    |     on aura INDLABEL = array([2, 3, 2, 1, 3, 1, 2])
    | label doit être de type tableau de chaine de caractères
    | labelnames doit être de type liste de chaine de caractères
    | INDLABEL est de type tableau de chaine de caractères
    '''
    nmlabel = list(labelnames);  # Copy pour préserver labelnames car ...
    if nmlabel[0] != '':  # ... si y'a pas '' devant on le met
        nmlabel.insert(0, '')
    nlabel = np.size(label);
    INDLABEL = np.ones(nlabel, dtype=np.int32) * -1;  # init
    for i in np.arange(nlabel):
        idx = nmlabel.index(label[i]);
        INDLABEL[i] = idx;
    return INDLABEL  # IND comme INDice (il s'agit donc d'un entier)


def mapclassif(sm, CBlabel, Data, bmus=None):
    # plm on doit passer les CBlabel (qui ne sont pas dans sm)
    ''' MAPCLASSE = mapclassif(sm, CBlabel, Data, bmus)
    | Classification de données par la carte (la carte est un classifieur)
    | En entrée :
    | sm      : La structure de la carte
    | CBlabel : Les labels des reférents (CodeBooks) qui peuvent être obtenus par
    |           la fonction cblabelmaj (sous forme de liste de string).
    | Data    : Les données auquelles ils convient d'attribuer un label (ce qui
    |           revient donc à les classer)
    | bmus    : les bmus sur lesquels on va s'appuyer pour affecter le label. Il
    |           doit s'agir des bmus associés aux données qui sont passées (Data).
    |           On ne vérifie que les données et les bmus ont la même taille. Si
    |           les bmus passés ne correspondent pas aux données la classification
    |           sera fausse, mais on ne peut pas le contrôler. Par défaut, on
    |           recherchera les bmus des données passées.
    |           La classification ne consiste en fait qu'à attribuer le label du
    |           bmus de la donnée.
    | En sortie :
    | MAPCLASSE : Les labels attribués aux données (sous forme de liste de string)
    '''
    nbdata = np.size(Data, 0);
    if bmus == [] or bmus is None:
        bmus = findbmus(sm, Data)
    else:
        if np.size(bmus) != nbdata:
            print("mapclassif : bmus and data must have the same number of elements");
            sys.exit(0);

    MAPCLASSE = [];
    for i in np.arange(nbdata):
        MAPCLASSE.append(CBlabel[int(bmus[i])]);
    return MAPCLASSE


def classifperf(sm, Xapp, Xapplabels, Xtest=None, Xtestlabels=None):
    ''' Performance en classification d'une carte topologique (sm)
    pour des données passées en paramètre. Les Labels des référents
    étant déterminés par vote majoritaire sur les données d'apprentissage
    '''
    if len(Xapp) != len(Xapplabels):
        print("classifperf: Il doit y avoir autant de labels que de données")
        sys.exit(0);
    if Xtest is None:
        Xtest = np.copy(Xapp);
    if Xtestlabels is None:  # si app et test on la meme taille
        Xtestlabels = np.copy(Xapplabels);  # on ne verra pas la supercherie
    if len(Xapp) != len(Xapplabels):
        print("classifperf: Il doit y avoir autant de labels que de données")
        sys.exit(0);
    if len(Xtest) != len(Xtestlabels):
        print("classifperf: Il doit y avoir autant de labels que de données")
        sys.exit(0);

    Tfreq, Ulab = reflabfreq(sm, sm.data, Xapplabels);
    CBlabmaj = cblabvmaj(Tfreq, Ulab);
    bmus = mbmus(sm, Xtest, narg=1);
    A = np.array(CBlabmaj)[[bmus[:, 0]]]
    Ig = np.where(A == Xtestlabels)[0]
    perf = len(Ig) / len(Xtest)
    return perf


def confus(sm, Data, Labels, classnames, CBlabels=None, Databmus=None, visu=False):
    #
    # Labelisation des référents (CB pour CodeBook) par vote majoritaire
    if CBlabels == None:  # j'ai rajouté ce paramètre pour éventuellement ne pas
        CBlabels = cblabelmaj(sm, Labels);  # le refaire à chaque fois

    # Bmus sur l'ensemble des données
    if Databmus is None:  # j'ai ajouté ce parmètre pour éventuellement ne pas
        Databmus = findbmus(sm, Data);  # le refaire à chaque fois

    # Classification (i.e labelisation) de l'ensemble des données par la carte
    Datamaplabs = mapclassif(sm, CBlabels, Data, Databmus);

    # Transformation des classes en int (because c'est ce qu'attend matconf)
    nmclasses = list(classnames);  # Copy pour préserver classname car j'ai vu
    # que ctk.label2ind ajoute '' devant ... (ce n'est plus vrai ?)
    Datailab = label2ind(Labels, nmclasses);
    Datamapilab = label2ind(Datamaplabs, nmclasses);
    #
    # Matrice de Confusion et Performance
    MC = tls.matconf(Datailab, Datamapilab, visu=visu);
    Perf = np.trace(MC) / np.sum(MC)
    if visu:  # Visualisation de la Matrice et Performance
        print("\n Matrice de Confusion")
        for i in np.arange(len(classnames)):  # equiv indice et label
            print('%d->%s  ' % (i + 1, classnames[i]), end='');
        print("Perf = %f" % Perf);
    return MC, Perf

# ========================= bottom of triedctk module ===========================
#
