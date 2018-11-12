cours(inf1005C).
cours(inf1010).
cours(inf1500).
cours(inf1600).
cours(inf2010).
cours(inf2705).
cours(inf1900).
cours(inf2205).
cours(mth1007).
cours(log2990).
cours(log1000).
cours(log2410).
cours(log2810).

corequis(log2810, inf2010).
corequis(log2990, inf2705).
corequis(mth1007, inf2705).
corequis(inf1900, inf2205).
corequis(inf1900, inf1600).
corequis(inf1900, log1000).

corequis(X, Y) :- corequis(Y, X).
corequis(X, Z) :- corequis(X, Y), corequis(Y, Z).

prerequis(inf1005C, log1000).
prerequis(inf1005C, inf1010).
prerequis(inf1005C, inf1600).
prerequis(inf1500, inf1600).
prerequis(inf1010, inf2010).
prerequis(inf1010, log2410).
prerequis(log1000, log2410).
prerequis(inf2010, inf2705).
getprerequis(A, B) :- prerequis(A, B).
getprerequis(X, Y) :- prerequis(X, Z), getprerequis(Z, Y). 

coursAPrendreComplet(A, Y) :- prerequis(Y, A); corequis(Y, A). 
