VMD write PDB with only proteins atoms (no HET atoms and no alternative locations):

protein and (altloc '' or altloc A)

MSMS command:

.\msms.exe -if .\1rwe-nohet-noalt.xyzrn -density 3.0 -of .\1rwe-high

