function [accuracy, precision, recall, F1] = ACCURACY_c(w, nometest)
%
% Lettura del test set contenente esempi (d) e relative etichette (c) esatte
%
load(nometest); 
examples = d'; 
labels = c;
n = length(labels);
%
% classificazione esempi del test set
%
b = w(1:end-1)'*examples + w(end);
%
% determinazione indici esempi negativi e positivi presenti nel test set
%
label_neg = find(sign(labels)<0);
label_pos = find(sign(labels)>0);
fprintf('\n numero esempi positivi nel testset %i\n',length(label_pos));
fprintf('\n numero esempi negativi nel testset %i\n',length(label_neg));
%
% calcolo di 
%
FN = nnz(b(label_pos)<0);
FP = nnz(b(label_neg)>0);
TP = length(label_pos)-FN;
TN = length(label_neg)-FP;
fprintf('\n TP=%i   TN=%i   FP=%i   FN=%i\n',TP, TN, FP, FN);
accuracy = (TP+TN)/n;
precision = (TP)/(TP+FP);
recall = (TP)/(TP+FN);
F1=2*(precision*recall)/(precision+recall);
