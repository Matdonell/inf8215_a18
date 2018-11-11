% ***********************************************************
% File: Exercice2_version_1.mzn  
% Name: Exercice 2: Round-Robin
% Desc: TODO
% Authors: Tanga Mathieu Kaboré, Reph ...
% Last update: 11/11/2018
% ***********************************************************

include "alldifferent.mzn";

int: nbTeams;
array[1..nbTeams, 1..nbTeams] of 0..1 : pv;

set of int: date = 0..nbTeams - 1;
set of int: datebis = 1..nbTeams - 4;
set of int: equipe = 1..nbTeams;

array[1..nbTeams, 1..nbTeams] of var date: matchs;

% Contrainte pour éliminer les matchs entre une même équipe.
constraint forall (i in equipe) (matchs[i, i] = 0);

% Contrainte pour garantir qu'une équipe joue une et une seule foi contre chacune des équipes.
constraint forall (i in equipe) (alldifferent( [ matchs[i, j] | j in equipe]) );

% 
% constraint forall (i, j in equipe) (matchs[i, j] = matchs[j, i]);

constraint forall (i, j, k, l, m in equipe)
  (if matchs[i, j] = matchs[i, k] + 1 /\
      matchs[i, j] = matchs[i, l] + 2 /\
      matchs[i, j] = matchs[i, m] + 3
   then 
     pv[i, j] != pv[i, k] \/ 
     pv[i, j] != pv[i, l] \/
     pv[i, j] != pv[i, m] 
   else 
     true
   endif);

solve satisfy;

output["\nCalendrier des matchs: \n\n"++ show2d(matchs)++ "\n\n"]