clear;
%
% Lettura dei parametri x_BB1_c che definiscono la funzione di predizione
% I parametri sono stati calcolati risolvendo un problema di ottimizzazione
%
load('pesi_BB1_c'); x = x_BB1_c; 
%load('pesi_SD'); x = x_s;
%
% Valutazione delle prestazioni della funzione di predizione su un test set
%
[accuracy, precision, recall, F1] = ACCURACY_c(x, 'test_w8a.mat');
fprintf('\n Accuracy:   (TP+TN)/n = %6.3f',accuracy);
fprintf('\n Precision: TP/(TP+FP) = %6.3f',precision);
fprintf('\n Recall:    TP/(TP+FN) = %6.3f',recall);
fprintf('\n F1:    2*(precision*recall)/(precision+recall) = %6.3f\n', F1);