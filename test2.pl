cours(iNF1005C).
cours(iNF1010).
cours(iNF1500).
cours(iNF1600).
cours(iNF2010).
cours(iNF2705).
cours(iNF1900).
cours(iNF2205).
cours(mTH1007).
cours(lOG2990).
cours(lOG1000).
cours(iNF2410).
cours(lOG2810).

corequis(lOG2810, iNF2010).
corequis(lOG2990, iNF2705).
corequis(mTH1007, iNF2705).
corequis(iNF1900, iNF2205).
corequis(iNF1900, iNF1600).
corequis(iNF1900, lOG1000).

corequis(X, Y):-corequis(Y, X).
corequis(X, Z):- corequis(X, Y),corequis(Y, Z).

prerequis(iNF1005C, lOG1000).
prerequis(iNF1005C, iNF1010).
prerequis(iNF1005C, iNF1600).
prerequis(iNF1500, iNF1600).
prerequis(iNF1010, iNF2010).
prerequis(iNF1010, lOG2410).
prerequis(lOG1000, lOG2410).
prerequis(iNF2010, iNF2705).
getprerequis(A, B):- prerequis(A, B).

getprerequis(X, Y):- prerequis(X, Z) , getprerequis(Z, Y). 

coursAPrendreComplet(A, Y):- prerequis(Y, A) ; corequis(Y, A). 
