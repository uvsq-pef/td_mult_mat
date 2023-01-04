set title 'Compteur energy-pkg en fonction de la taille de la matrice'
set xlabel 'Taille de matrice'
set ylabel 'Consommation (Joules)'

set term svg
set output 'energy-pkg.svg'

set datafile separator ';'
plot "energy-pkg.dat" using 1:2 title 'energy-pkg' with linespoints
