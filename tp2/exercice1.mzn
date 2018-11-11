% ***********************************************************
% File: Exercice1.mzn  
% Name: Détective
% Desc: TODO
% Authors: Tanga Mathieu Kaboré, Reph ...
% Last update: 11/11/2018
% ***********************************************************

include "alldifferent.mzn";
int: pos=5;

% Représentation des couleurs
var 1..pos: rouge; 
var 1..pos: vert; 
var 1..pos: blanc; 
var 1..pos: jaune;
var 1..pos: bleu; 

% Représentation des nationalités
var 1..pos: anglais;
var 1..pos: espagnol; 
var 1..pos: ukrainien; 
var 1..pos: norvegien; 
var 1..pos: japonais;

% Représentation des animaux
var 1..pos: chien; 
var 1..pos: escargot; 
var 1..pos: renard; 
var 1..pos: cheval; 
var 1..pos: zebre; 

var 1..pos: sculpteur; 
var 1..pos: diplomate; 
var 1..pos: violoniste; 
var 1..pos: medecin; 
var 1..pos: acrobate; 

% Représentation des boissons
var 1..pos: cafe; 
var 1..pos: the; 
var 1..pos: lait;
var 1..pos: jus; 
var 1..pos: eau;

% Définition des contraintes d'unicité
constraint alldifferent([rouge, vert, blanc, jaune, bleu]);
constraint alldifferent([anglais, espagnol, ukrainien, norvegien, japonais]);
constraint alldifferent([chien, escargot, renard, cheval, zebre]);
constraint alldifferent([sculpteur, diplomate, violoniste, medecin, acrobate]);
constraint alldifferent([cafe, the, lait, jus, eau]);

% Définition des contraintes de relation
constraint anglais = rouge;
constraint espagnol = chien;
constraint vert = cafe;
constraint ukrainien = the;
constraint vert = blanc + 1;
constraint sculpteur = escargot;
constraint diplomate = jaune;
constraint lait = 3;
constraint norvegien = 1;
constraint medecin = renard + 1 \/ medecin = renard - 1;
constraint diplomate = cheval + 1 \/ diplomate = cheval - 1;
constraint violoniste = jus;
constraint japonais = acrobate;
constraint norvegien = bleu + 1 \/ norvegien = bleu - 1;

solve satisfy;