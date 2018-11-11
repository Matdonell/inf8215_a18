% ***********************************************************
% File: Exercice1.mzn  
% Name: Détective
% Desc: TODO
% Authors: Tanga Mathieu Kaboré, Reph ...
% Last update: 11/11/2018
% ***********************************************************

include "alldifferent.mzn";

int: pos = 5;
set of int: position = 1..pos;

% Représentation des maisons en couleurs
var position: rouge; 
var position: vert; 
var position: blanc; 
var position: jaune;
var position: bleu; 

% Représentation des nationalités
var position: anglais;
var position: espagnol; 
var position: ukrainien; 
var position: norvegien; 
var position: japonais;

% Représentation des animaux
var position: chien; 
var position: escargot; 
var position: renard; 
var position: cheval; 
var position: zebre; 

% Représentation des professions des individus
var position: sculpteur; 
var position: diplomate; 
var position: violoniste; 
var position: medecin; 
var position: acrobate; 

% Représentation des boissons
var position: cafe; 
var position: the; 
var position: lait;
var position: jus; 
var position: eau;

array[int] of var int: nationalites = [anglais, espagnol, ukrainien, norvegien, japonais];
% Cette liste est utilisée pour retrouver et afficher les personnes en format texte après la résolution
% Elle doit conserver l'ordre de la liste des nationalités ci-dessus 
array[position] of string: nationalites_name = ["anglais", "espagnol", "ukrainien", "norvegien", "japonais"];

array[int] of var int: maisons = [rouge, vert, blanc, jaune, bleu];
array[int] of var int: animaux = [chien, escargot, renard, cheval, zebre];
array[int] of var int: professions = [sculpteur, diplomate, violoniste, medecin, acrobate];
array[int] of var int: boissons = [cafe, the, lait, jus, eau];

% Définition des contraintes d'unicité
constraint alldifferent(maisons);
constraint alldifferent(nationalites);
constraint alldifferent(animaux);
constraint alldifferent(professions);
constraint alldifferent(boissons);

% Définition des contraintes de relation

% L’Anglais habite à la maison rouge.
constraint anglais = rouge;

% L’Espagnol a un chien.
constraint espagnol = chien;

% Dans la maison verte, on boit du café.
constraint vert = cafe;

% L’Ukrainien boit du thé.
constraint ukrainien = the;

% La maison verte est immédiatement à droite de la maison blanche.
constraint vert = blanc + 1;

% Le sculpteur élève des escargots.
constraint sculpteur = escargot;

% Le diplomate habite la maison jaune.
constraint diplomate = jaune;

% Dans la maison du milieu, on boit du lait.
constraint lait = 3;

% Le Norvégien habite à la première maison à gauche.
constraint norvegien = 1;

% Le médecin habite dans une maison voisine de celle où demeure le propriétaire du renard.
constraint medecin = renard + 1 \/ medecin = renard - 1;

% La maison du diplomate est à côté de celle où il y a un cheval.
constraint diplomate = cheval + 1 \/ diplomate = cheval - 1;

% Le violoniste boit du jus d’orange.
constraint violoniste = jus;

% Le Japonais est acrobate.
constraint japonais = acrobate;

% Le Norvégien habite à côté de la maison bleue.
constraint norvegien = bleu + 1 \/ norvegien = bleu - 1;


solve satisfy;


% Affichage du résultat final
output ["\nSolution 1: L'" ++ nationalites_name[fix(g)]++
        " boit de l'eau\n" | g in nationalites where fix(g)==fix(eau)];
        
output ["\nSolution 2: Le " ++ nationalites_name[fix(g)]++
        " possède le zèbre\n\n" | g in nationalites where fix(g)==fix(zebre)];

