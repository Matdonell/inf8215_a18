% Use this editor as a MiniZinc scratch book
include "alldifferent.mzn";
int: N=14;
set of int: date= 1..N-1;
set of int: datebis= 1..N-4;
set of int: equipe=1..14;
array[1..N,1..N-1] of var equipe: calendrier;
constraint forall (i in equipe, j in date) (calendrier[i, j] !=i);
constraint forall (j in date)(alldifferent([calendrier[i, j] | i in equipe]));
constraint forall(i in equipe)(alldifferent([calendrier[i, j] | j in date]));
constraint forall (i in equipe, j in datebis) (pv[i,calendrier[i, j]]!=pv[i, calendrier[i, j+1]] \/
pv[i,calendrier[i, j]]!=pv[i, calendrier[i, j+2]]\/
 pv[i,calendrier[i, j]]!=pv[i,calendrier[i, j+3]]);

array[1..14, 1..14] of 0..1 : pv = [|
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1|
1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1|
0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0|
0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0|
1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0|
1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0|
0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1|
1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0|
1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0|
1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1|
0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0|
0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0|
0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0|
0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0|];

solve satisfy;
