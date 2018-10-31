% Use this editor as a MiniZinc scratch book
include "alldifferent.mzn";
int: N=14;
set of int: date= 0..N-1;
set of int: datebis= 1..N-4;
set of int: equipe=1..14;
set of array[1..1,1..2]: match=[i, j]|i in equipe, j in date;
array[1..N,1..N-1] of var match: calendrier;
constraint forall(i in equipe, j in date) (calendrier[i, j][1, 1]=i); 
constraint forall (i in equipe, j in date) (alldifferent([calendrier[i, j][1, 1], calendrier[i, j][1, 2]]));
constraint forall (j in date) (alldifferent([calendrier[i, j] | i in equipe]));
constraint forall(i,k in equipe, j in date) (if calendrier[i, j][1, 1]=calendrier[i, k][1, 2] then calendrier[i, k][1, 1]=calendrier[i, j][1, 2] else true endif);

constraint forall (i in equipe, j in datebis) (pv[i,j]!=pv[i,j+1] \/pv[i,j]!=pv[i, j+2] \/pv[i,j]!=pv[i, j+3]);
 

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
