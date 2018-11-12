% Liste des personnes :
% ---------------------

homme(michael_jackson).
homme(hideo_kojima).
homme(banksy).
homme(mario).
homme(quentin_tarantino).
homme(joseph_staline).
homme(dwight_eisenhower).
homme(mikhail_gorbachev).
homme(victor_hugo).
homme(jesus).
homme(ayrton_senna).
homme(moise).
homme(fernando_alonso).
homme(pape_francois).
homme(james_bond).
homme(denzel_washington).
homme(richard_nixon).

femme(lady_gaga).
femme(jennifer_lawrence).
femme(lara_croft).
femme(cleopatre).
femme(j_k_rowling).

personne(X) :-
    femme(X); homme(X).

film(james_bond).
film(lara_croft).
film(jennifer_lawrence).
film(quentin_tarantino).
film(denzel_washington).
film(richard_nixon).

jeu_video(mario).
jeu_video(lara_croft).
jeu_video(hideo_kojima).

reel(michael_jackson).
reel(hideo_kojima).
reel(banksy).
reel(j_k_rowling).
reel(quentin_tarantino).
reel(joseph_staline).
reel(dwight_eisenhower).
reel(mikhail_gorbachev).
reel(victor_hugo).
reel(jesus).
reel(ayrton_senna).
reel(moise).
reel(fernando_alonso).
reel(pape_francois).
reel(denzel_washington).
reel(richard_nixon).
reel(lady_gaga).
reel(jennifer_lawrence).
reel(cleopatre).

en_vie(hideo_kojima).
en_vie(lady_gaga).
en_vie(jennifer_lawrence).
en_vie(j_k_rowling).
en_vie(quentin_tarantino).
en_vie(fernando_alonso).
en_vie(pape_francois).
en_vie(denzel_washington).

politicien(mikhail_gorbachev).
politicien(joseph_staline).
politicien(dwight_einsenhower).
politicien(cleopatre).
politicien(victor_hugo).
politicien(moise).
politicien(pape_francois).
politicien(richard_nixon).

religieux(jesus).
religieux(moise).
religieux(pape_francois).

vieux(michael_jackson).
vieux(joseph_staline).
vieux(dwight_eisenhower).
vieux(mikhail_gorbachev).
vieux(victor_hugo).
vieux(jesus).
vieux(moise).
vieux(pape_francois).
vieux(james_bond).
vieux(denzel_washington).
vieux(richard_nixon).
vieux(cleopatre).

tafta(michael_jackson).
tafta(jennifer_lawrence).
tafta(banksy).
tafta(lara_croft).
tafta(mario).
tafta(j_k_rowling).
tafta(lady_gaga).
tafta(quentin_tarantino).
tafta(victor_hugo).
tafta(dwight_einsenhower).
tafta(richard_nixon).
tafta(james_bond).
tafta(denzel_washington).

acteur(jennifer_lawrence).
acteur(denzel_washington).
musicien(lady_gaga).
musicien(michael_jackson).
artiste(j_k_rowling).
artiste(banksy).
artiste(victor_hugo).
artiste(X) :- (musicien(X); jeu_video(X); film(X)),reel(X).


ask(acteur, Y) :-
    format('~w est un acteur ? ', [Y]),
    read(Reponse),
    Reponse = 'oui'.
ask(musicien, X) :-
    format('~w est un musicien ? ', [X]),
    read(Reponse),
    Reponse = 'oui'.
ask(artiste, X) :-
    format('~w est un artiste ? ', [X]),
    read(Reponse),
    Reponse = 'oui'.


% Objets :
objet(aspirateur).
objet(ordinateur).
objet(telephone).
objet(fourchette).
objet(balai).
objet(cactus).
objet(assiette).
objet(four).
objet(cuisiniere).
objet(cafetiere).
objet(grille_pain_table).
objet(casserole).
objet(shampooing).
objet(detergent_a_vaisselle).
objet(lit).
objet(cle).
objet(portefeuille).
objet(sac_a_dos).
objet(piano).
objet(lampe).
objet(papier).
