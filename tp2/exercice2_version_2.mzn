% **************************************************************************
% File: Exercice2_version_2.mzn  
% Name: Exercice 2: Round-Robin
% Desc: Constraint programming to resolve a Round-Robin Tournament problem
% Authors: Tanga Mathieu Kaboré, Florence Gaborit, Reph dauphin Mombrun
% Last update: 11/11/2018
% **************************************************************************

include "alldifferent.mzn";

int: nbTeams;
array[1..nbTeams, 1..nbTeams] of 0..1 : pv; % Location des matchs (exterieur ou domicile)

set of int: date = 1..nbTeams - 1;

% On a un total de 14 équipes, utilisé pour la limitation de matchs successifs (1 à 10)
set of int: datebis = 1..nbTeams - 4; 
set of int: equipe = 1..nbTeams;

% Tableau du calendrier à 2 dimensions
array[1..nbTeams, 1..nbTeams - 1] of var equipe: calendrier;

constraint forall (i in equipe, j in date) (calendrier[i, j] != i);

% Contrainte d'unicité sur les matchs par date
constraint forall (j in date)(alldifferent([calendrier[i, j] | i in equipe]));

% Contrainte d'unicité sur les dates par matchs
constraint forall (i in equipe)(alldifferent([calendrier[i, j] | j in date]));

% Contrainte sur le nombre de matchs successifs limité à 4
constraint forall (i in equipe, j in datebis) (
    pv[i, calendrier[i, j]] != pv[i, calendrier[i, j + 1]] \/
    pv[i, calendrier[i, j]] != pv[i, calendrier[i, j + 2]]\/
    pv[i, calendrier[i, j]] != pv[i, calendrier[i, j + 3]]);

solve satisfy;

output["\nCalendrier des matchs: \n\n"++ show2d(calendrier)++ "\n\n"]