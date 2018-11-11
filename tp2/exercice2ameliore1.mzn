% ***********************************************************
% File: Exercice2.mzn  
% Name: Exercice 2: Round-Robin
% Desc: TODO
% Authors: Tanga Mathieu Kaboré, Reph ...
% Last update: 11/11/2018
% ***********************************************************

include "alldifferent.mzn";

int: nbTeams;
array[1..nbTeams, 1..nbTeams] of 0..1 : pv;

set of int: date = 1..nbTeams - 1;
set of int: datebis = 1..nbTeams - 4;
set of int: equipe = 1..nbTeams;

array[1..nbTeams, 1..nbTeams-1] of var equipe: calendrier;

constraint forall (i in equipe, j in date) (calendrier[i, j] !=i);
constraint forall (j in date)(alldifferent([calendrier[i, j] | i in equipe]));
constraint forall(i in equipe)(alldifferent([calendrier[i, j] | j in date]));
constraint forall (i in equipe, j in datebis) (pv[i,calendrier[i, j]]!=pv[i, calendrier[i, j+1]] \/
pv[i,calendrier[i, j]]!=pv[i, calendrier[i, j+2]]\/
 pv[i,calendrier[i, j]]!=pv[i,calendrier[i, j+3]]);

solve satisfy;