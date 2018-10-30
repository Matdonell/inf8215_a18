% Use this editor as a MiniZinc scratch book
include "alldifferent.mzn";
int: N=14;
set of int: date= 0..N-1;
set of int: datebis= 1..N-4;
set of int: equipe=1..14;
array[1..N,1..N] of var date: matchs;
constraint forall (i in equipe) (matchs[i, i]=0);
constraint forall (i in equipe) (alldifferent( [ matchs[i,j] | j in equipe]) );
constraint forall(i,j in equipe) (matchs[i,j]=matchs[j,i]);

constraint forall (i, j, k, l, m in equipe) (if matchs[i, j]=matchs[i, k]+1 /\ matchs[i, j]=matchs[i, l]+2 /\ matchs[i, j]=matchs[i, m]+3 then pv[i,j]!=pv[i,k] \/pv[i,j]!=pv[i, l] \/pv[i,j]!=pv[i, m] else true endif);
 

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
