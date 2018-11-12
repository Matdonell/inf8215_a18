% **************************************************************************
% File: Exercice2_version_1.mzn  
% Name: Exercice 2: Round-Robin
% Desc: Constraint programming to resolve a Round-Robin Tournament problem
% Authors: Tanga Mathieu Kaboré, Florence Gaborit, Reph dauphin Mombrun
% Last update: 11/11/2018
% **************************************************************************

include "alldifferent.mzn";

int: nbTeams;
array[1..nbTeams, 1..nbTeams] of 0..1 : pv;

set of int: date = 0..nbTeams - 1;
set of int: datebis = 1..nbTeams - 4;
set of int: equipe = 1..nbTeams;

array[1..nbTeams, 1..nbTeams] of var date: matchs;

% Contrainte pour éliminer les matchs entre une même équipe.
constraint forall (i in equipe) (matchs[i, i] = 0);

% Contrainte pour garantir des matchs différents entre les équipes
constraint forall (i in equipe) (alldifferent( [ matchs[i, j] | j in equipe]) );

% Contrainte pour éliminer les relations symétriques entre les matchs
constraint forall (i, j in equipe) (matchs[i, j] = matchs[j, i]);

% Contrainte sur le nombre de matchs
constraint forall (i, j, k, l, m in equipe)
  (if matchs[i, j] = matchs[i, k] + 1 /\ % Contrainte sur le 1e match suivant
      matchs[i, j] = matchs[i, l] + 2 /\ % Contrainte sur le 2e match suivant
      matchs[i, j] = matchs[i, m] + 3 % Containte sur le 3è match successif
   then 
     pv[i, j] != pv[i, k] \/ 
     pv[i, j] != pv[i, l] \/
     pv[i, j] != pv[i, m] 
   else 
     true
   endif);

solve satisfy;

output["\nCalendrier des matchs: \n\n"++ show2d(matchs)++ "\n\n"]