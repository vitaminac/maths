* Una empresa elabora un cierto alimento
* refinando diferentes tipos de aceite y mezclandolos
* Los tipos de aceite se clasifican en dos categorias
* vegetales (VEG1 y VEG2) y no vegetales (OIL1, OIL2 Y OIL3)
Sets
         i aceites /VEG1, VEG2, OIL1, OIL2, OIL3/
;

* La capacidad maxima de refino es de
* 200 Toneladas de aceite vegetal
Scalar max_vegetal /200/;
* Y 250 Tonelada de no vegetal
Scalar max_no_vegetal /250/;

* Por restricciones tecnologica
* la dureza del producto final
* debe estar entre 3 y 6 unidades
Scalar min_dureza /3/;
Scalar max_dureza /6/;

* Se puede asumir que la dureza se mezcla linealmente
* La dureza final es la media ponderada
* de la dureza de los distintos aceites utilizados en la mezcla
Parameters
         coste(i)
       / VEG1       110
         VEG2       120
         OIL1       130
         OIL2       110
         OIL3       115 /

         dureza
       / VEG1       8.8
         VEG2       6.1
         OIL1       2.0
         OIL2       4.2
         OIL3       5.0 /
;

* Cada tonelada de producto final se vende a un precio de 150
Scalar beneficio /150/;

Free Variables
         z beneficio total
;

Positive Variables
         x(i) cantidad de producto i
;

Equations
         obj beneficio total
         r_max_vegetal
         r_max_no_vegetal
         r_min_dureza
         r_max_dureza
;

obj.. z =e= sum(i, beneficio*x(i)) - sum(i, coste(i)*x(i));
r_max_vegetal.. max_vegetal =g= x('VEG1') + x('VEG2');
r_max_no_vegetal.. max_no_vegetal =g= x('OIL1') + x('OIL2') + x('OIL3');
r_min_dureza.. min_dureza =l= sum(i,x(i)*dureza(i));
r_max_dureza.. max_dureza =g= sum(i,x(i)*dureza(i));
Model aceite /ALL/;
Solve aceite using LP maximizing z;