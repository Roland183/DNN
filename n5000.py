#!/usr/bin/python3
#----------------------------------------------------------------
# Mein Dynamisches Neuronales Netz in github2
# Dateiname: 2028_ein_DNN_Areal_org.py
# R.J.Nickerl mit github
# 05.04.20 Python 3.8
#--------------------------------------------------------------
try:
    import tkinter as tk
except ImportError:
    import Tkinter as tk                        
from random import randint
import numpy
import math
import scipy.special
import matplotlib.pyplot as plt
#import math
#import time
#import RPi.GPIO as GPIO
#import datetime

# Dynamisches Neuronales Netz class definition
class dynNN1:
    def __init__(self,inputnodes, hiddennodes, outputnodes):
        # initialise the dynNN
        # set number of nodes in each input, hidden, output and daempfungs layer
        self.inodes1 = inputnodes        # Anzahl Input Nodes
        self.hnodes1 = hiddennodes       # Anzahl Hidden Nodes
        self.dnodes1 = self.hnodes1      # Anzahl Damp Nodes (=Dämpfungsknoten)
        self.onodes1 = outputnodes       # Anzahl Output Nodes
        
        # link weight matrices: wih, who and dhh
        # weights inside the arrays are:
        # w_i_j, where link is from node i to node j in the next layer
        # d_i_j, where reverse link is from node i to node j in the same layer
        # w11 w21   d11 d21
        # w12 w22   d12 d22
        self.wih1 = numpy.random.normal(0.0, pow(self.hnodes1, -0.5), (self.hnodes1, self.inodes1))
        self.who1 = numpy.random.normal(0.0, pow(self.onodes1, -0.5), (self.onodes1, self.hnodes1))
        self.dihh1 = numpy.random.normal(0.0, pow(self.dnodes1, -0.5), (self.hnodes1, self.hnodes1))
        self.dohh1 = numpy.random.normal(0.0, pow(self.dnodes1, -0.5), (self.onodes1, self.onodes1))
        pass
    
        # Vector: Dihh[i] self.hidden_outputs werden random erstbesetzt
        # Vector: Dohh[i] self.output_outputs werden random erstbesetzt
        self.hidden_outputs1 = numpy.random.normal(0.0, pow(self.hnodes1, -0.5), (self.hnodes1))
        self.output_outputs1 = numpy.random.normal(0.0, pow(self.hnodes1, -0.5), (self.onodes1))
        #print("Dihh[i]= ", self.hidden_outputs)
        #print("Dohh[i]= ", self.output_outputs)
        print("++++++++++++++++ DNN 1 ++++++++++++++++")
        print("Dihh1[i]= ", self.hidden_outputs1.round(3))
        print("Dohh1[i]= ", self.output_outputs1.round(3))


        
        # acivation function is the sigmoid function Änderung: hier wird die Stufenfunktion verwendet
        self.activation_function = lambda x: scipy.special.expit(x)

    
    def status1(self):
        # status dynNN
        #print("Anzahl input nodes= ", self.inodes+1)
        #print("Anzahl hidden nodes= ", self.hnodes+1)
        #print("Anzahl output nodes= ", self.onodes+1)
        print("++++++++++++++++ DNN 1 ++++++++++++++++")
        print("wih1= ")
        print(self.wih1.round(3))
        print("who1= ")
        print(self.who1.round(3))
        print("dihh1= ")
        print(self.dihh1.round(3))
        print("dohh1= ")
        print(self.dohh1.round(3))

        #print("Dihh[i]= ", self.hidden_outputs.round(3))
        #print("Dohh[i]= ", self.output_outputs.round(3))
        print()         
        pass

    
    def search(self, inputs_list):
        # search dynNN
        inputs = inputs_list
        print("..........SUCHE...............")
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih1, inputs)
        print(self.wih, inputs)
        pass

    def vektor_hi1(self, inputs_list):
        # Schritt 6: Vektor hi berechnen: hi = wih @ INPUT_akt[i]-Vektor + dihh @ self.hidden_outputs
        ##print("Schritt 6: Vektor hi berechnen")
        inputs = inputs_list
        #print("xxxxxclass hidden_outputs VORHER= \n{}".format(self.hidden_outputs.round(3)))
        self.hidden_inputs1 = numpy.dot(self.wih1, inputs) + numpy.dot(self.dihh1, self.hidden_outputs1)
        #print("....class numpy.dot(self.wih, inputs)= \n{}".format(numpy.dot(self.wih, inputs.round(3))))
        #print("....class numpy.dot(self.dihh, self.hidden_outputs)= \n{}".format(numpy.dot(self.dihh, self.hidden_outputs.round(3))))
        #print("....class hidden_inputs= \n{}".format(self.hidden_inputs.round(3)))
        self.hidden_outputs1=self.hidden_inputs1
        #print("xxxxxclass hidden_outputs NACHHER= \n{}".format(self.hidden_outputs.round(3)))
        return self.hidden_outputs1
        pass
    
    def sprung_antwort_hidden1(self, stufenwert):
        # Schritt 7: Sprungantwort der hidden Neuronen hs (s=Sprung) entspr. dem Stufenwer (stufen_wert) berechnen
        ##print("Schritt 7: Sprungantwort der hidden Neuronen")
        self.swert1 = stufenwert # Höhe des Schwellwertes der Nodes bei dem der Knoten "schaltet"
        for i in range(self.hnodes1):
            if self.hidden_outputs1[i] > self.swert1:
                self.hidden_outputs1[i] = 1
            else:
                self.hidden_outputs1[i] = 0
            ##print("self.hidden_outputs[i]xxx= ", self.hidden_outputs[i], i)
            #print("self.swert xxx= ", self.swert, "  ...", i)            
        return self.hidden_outputs1
        pass    

    def vektor_ho1(self):
        # Schritt 9: Vektor ho berechnen: ho = who @ self.hidden_outputs-Vektor + dohh @ self.output_outputs
        ##print("Schritt 9: Vektor ho berechnen")
        #print("xxxxxclass output_outputs VORHER= \n{}".format(self.outputs_outputs.round(3)))
        self.output_inputs1 = numpy.dot(self.who1, self.hidden_outputs1) + numpy.dot(self.dohh1, self.output_outputs1)
        #print("....class numpy.dot(self.who, inputs)= \n{}".format(numpy.dot(self.who, self.hidden_outputs).round(3)))
        #print("....class numpy.dot(self.who, inputs)= ")
        #print(numpy.dot(self.who, self.hidden_outputs))
        #print("....class numpy.dot(self.dohh, self.hidden_outputs)= \n{}".format(numpy.dot(self.dohh, self.output_outputs).round(3)))
        #print("....class output_inputs= \n{}".format(self.output_inputs.round(3)))
        self.output_outputs1=self.output_inputs1
        #print("xxxxxclass hidden_outputs NACHHER= \n{}".format(self.hidden_outputs.round(3)))
        return self.output_outputs1
        pass

    def sprung_antwort_output1(self, stufenwert):
        # Schritt 10: Sprungantwort der output Neuronen o1 - on entspr. dem Stufenwert berechnen
        ##print("Schritt 9: Sprungantwort der output Neuronen")
        self.swert1 = stufenwert # Höhe des Schwellwertes der Nodes bei dem der Knoten "schaltet"
        for i in range(self.onodes1):
            if self.output_outputs1[i] > self.swert1:
                self.output_outputs1[i] = 1
            else:
                self.output_outputs1[i] = 0
            ##print("<<<<self.output_outputs[i]xxx= ", self.output_outputs[i], i)
            #print("self.swert xxx= ", self.swert, "  ...", i)            
        return self.output_outputs1
        pass

    def daempf_anpassung1(self):
        # Ziel ist ein chaotisches Schwingen (0,1,0,1,0,1,...) zu detektieren und zum Zeitpkt.=0 zu dämpfen
        # Dämpfungsnodes um 1 = einen Zeitpunkt verkleinern
        # Dämpfungswerte dhh anpassen. 
        # Dämpfungsmatrix erzeugen wenn Zeitwert z=0 dann dämpfen   
        print("ccccccccc daempfung_anpassung1 ccccccccc")
        #print("hs vorher= \n{}".format(self.hs))
        #print("dhh vorher= \n{}".format(self.dhh.round(2)))        
        for i in range(hidden_nodes):
            for k in range(hidden_nodes):
                if self.hs_alt1[i] != self.hs1[i]:
                    self.dhh1[i,k] = 1.25*self.dhh1[i,k]  # Detektion von 0,1,0,.. Feuern des Neurons hs[i]=>Chaos=>Erhöhung der Dämpfung          
                    #print("hsalt ungl hsneu")
                    #print("self.hidden_inputs[i]= ", self.hidden_inputs[i], i)
                    #print("self.dhh[i,k]= ", self.dhh[i,k], i, k)
                    pass
                if self.hs_alt1[i] == self.hs1[i]:
                    self.dhh1[i,k] = 0.75*self.dhh1[i,k]  # Detektion von 0,0,0, oder 1,1,1.. kein/immer Feuern des Neurons => Dämpfung auf 75% verkleiner          
                    pass
                
        #print("hs nachher= \n{}".format(self.hs))
        #print("dhh nachher= \n{}".format(self.dhh.round(2)))
        return self.hs1
        pass

    
    # train dynNN

    # query dynNN
    
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class dynNN2:
    def __init__(self,inputnodes, hiddennodes, outputnodes):
        self.inodes2 = inputnodes        # Anzahl Input Nodes
        self.hnodes2 = hiddennodes       # Anzahl Hidden Nodes
        self.dnodes2 = self.hnodes2      # Anzahl Damp Nodes (=Dämpfungsknoten)
        self.onodes2 = outputnodes       # Anzahl Output Nodes       
        self.wih2 = numpy.random.normal(0.0, pow(self.hnodes2, -0.5), (self.hnodes2, self.inodes2))
        self.who2 = numpy.random.normal(0.0, pow(self.onodes2, -0.5), (self.onodes2, self.hnodes2))
        self.dihh2 = numpy.random.normal(0.0, pow(self.dnodes2, -0.5), (self.hnodes2, self.hnodes2))
        self.dohh2 = numpy.random.normal(0.0, pow(self.dnodes2, -0.5), (self.onodes2, self.onodes2))
        pass
        self.hidden_outputs2 = numpy.random.normal(0.0, pow(self.hnodes2, -0.5), (self.hnodes2))
        self.output_outputs2 = numpy.random.normal(0.0, pow(self.hnodes2, -0.5), (self.onodes2))
        print("++++++++++++++++ DNN 2 ++++++++++++++++")
        print("Dihh2[i]= ", self.hidden_outputs2.round(3))
        print("Dohh2[i]= ", self.output_outputs2.round(3))
    def status2(self):
        print("++++++++++++++++ DNN 2 ++++++++++++++++")
        print("wih2= ")
        print(self.wih2.round(3))
        print("who2= ")
        print(self.who2.round(3))
        print("dihh2= ")
        print(self.dihh2.round(3))
        print("dohh2= ")
        print(self.dohh2.round(3))
        print()         
        pass
    def search(self, inputs_list):
        inputs = inputs_list
        hidden_inputs = numpy.dot(self.wih2, inputs)
        print(self.wih, inputs)
        pass
    def vektor_hi2(self, inputs_list):
        inputs = inputs_list
        self.hidden_inputs2 = numpy.dot(self.wih2, inputs) + numpy.dot(self.dihh2, self.hidden_outputs2)
        self.hidden_outputs2=self.hidden_inputs2
        return self.hidden_outputs2
        pass    
    def sprung_antwort_hidden2(self, stufenwert):
        self.swert2 = stufenwert # Höhe des Schwellwertes der Nodes bei dem der Knoten "schaltet"
        for i in range(self.hnodes2):
            if self.hidden_outputs2[i] > self.swert2:
                self.hidden_outputs2[i] = 1
            else:
                self.hidden_outputs2[i] = 0
        return self.hidden_outputs2
        pass    
    def vektor_ho2(self):
        self.output_inputs2 = numpy.dot(self.who2, self.hidden_outputs2) + numpy.dot(self.dohh2, self.output_outputs2)
        self.output_outputs2=self.output_inputs2
        return self.output_outputs2
        pass
    def sprung_antwort_output2(self, stufenwert):
        self.swert2 = stufenwert # Höhe des Schwellwertes der Nodes bei dem der Knoten "schaltet"
        for i in range(self.onodes2):
            if self.output_outputs2[i] > self.swert2:
                self.output_outputs2[i] = 1
            else:
                self.output_outputs2[i] = 0
        return self.output_outputs2
        pass
    def daempf_anpassung2(self):
        for i in range(hidden_nodes):
            for k in range(hidden_nodes):
                if self.hs_alt2[i] != self.hs2[i]:
                    self.dhh2[i,k] = 1.25*self.dhh2[i,k]  # Detektion von 0,1,0,.. Feuern des Neurons hs[i]=>Chaos=>Erhöhung der Dämpfung          
                    pass
                if self.hs_alt2[i] == self.hs2[i]:
                    self.dhh2[i,k] = 0.75*self.dhh2[i,k]  # Detektion von 0,0,0, oder 1,1,1.. kein/immer Feuern des Neurons => Dämpfung auf 75% verkleiner          
                    pass
        return self.hs2
        pass

    
    # train dynNN

    # query dynNN

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class dynNN3:
    def __init__(self,inputnodes, hiddennodes, outputnodes):
        self.inodes3 = inputnodes        # Anzahl Input Nodes
        self.hnodes3 = hiddennodes       # Anzahl Hidden Nodes
        self.dnodes3 = self.hnodes3      # Anzahl Damp Nodes (=Dämpfungsknoten)
        self.onodes3 = outputnodes       # Anzahl Output Nodes       
        self.wih3 = numpy.random.normal(0.0, pow(self.hnodes3, -0.5), (self.hnodes3, self.inodes3))
        self.who3 = numpy.random.normal(0.0, pow(self.onodes3, -0.5), (self.onodes3, self.hnodes3))
        self.dihh3 = numpy.random.normal(0.0, pow(self.dnodes3, -0.5), (self.hnodes3, self.hnodes3))
        self.dohh3 = numpy.random.normal(0.0, pow(self.dnodes3, -0.5), (self.onodes3, self.onodes3))
        pass
        self.hidden_outputs3 = numpy.random.normal(0.0, pow(self.hnodes3, -0.5), (self.hnodes3))
        self.output_outputs3 = numpy.random.normal(0.0, pow(self.hnodes3, -0.5), (self.onodes3))
        print("++++++++++++++++ DNN 3 ++++++++++++++++")
        print("Dihh3[i]= ", self.hidden_outputs3.round(3))
        print("Dohh3[i]= ", self.output_outputs3.round(3))
    def status3(self):
        print("++++++++++++++++ DNN 3 ++++++++++++++++")
        print("wih3= ")
        print(self.wih3.round(3))
        print("who3= ")
        print(self.who3.round(3))
        print("dihh3= ")
        print(self.dihh3.round(3))
        print("dohh3= ")
        print(self.dohh3.round(3))
        print()         
        pass
    def search(self, inputs_list):
        inputs = inputs_list
        hidden_inputs = numpy.dot(self.wih3, inputs)
        print(self.wih, inputs)
        pass
    def vektor_hi3(self, inputs_list):
        inputs = inputs_list
        self.hidden_inputs3 = numpy.dot(self.wih3, inputs) + numpy.dot(self.dihh3, self.hidden_outputs3)
        self.hidden_outputs3 = self.hidden_inputs3
        return self.hidden_outputs3
        pass    
    def sprung_antwort_hidden3(self, stufenwert):
        self.swert3 = stufenwert # Höhe des Schwellwertes der Nodes bei dem der Knoten "schaltet"
        for i in range(self.hnodes3):
            if self.hidden_outputs3[i] > self.swert3:
                self.hidden_outputs3[i] = 1
            else:
                self.hidden_outputs3[i] = 0
        return self.hidden_outputs3
        pass    
    def vektor_ho3(self):
        self.output_inputs3 = numpy.dot(self.who3, self.hidden_outputs3) + numpy.dot(self.dohh3, self.output_outputs3)
        self.output_outputs3=self.output_inputs3
        return self.output_outputs3
        pass
    def sprung_antwort_output3(self, stufenwert):
        self.swert3 = stufenwert # Höhe des Schwellwertes der Nodes bei dem der Knoten "schaltet"
        for i in range(self.onodes3):
            if self.output_outputs3[i] > self.swert3:
                self.output_outputs3[i] = 1
            else:
                self.output_outputs3[i] = 0
        return self.output_outputs3
        pass
    def daempf_anpassung3(self):
        for i in range(hidden_nodes):
            for k in range(hidden_nodes):
                if self.hs_alt3[i] != self.hs3[i]:
                    self.dhh3[i,k] = 1.25*self.dhh3[i,k]  # Detektion von 0,1,0,.. Feuern des Neurons hs[i]=>Chaos=>Erhöhung der Dämpfung          
                    pass
                if self.hs_alt3[i] == self.hs3[i]:
                    self.dhh3[i,k] = 0.75*self.dhh3[i,k]  # Detektion von 0,0,0, oder 1,1,1.. kein/immer Feuern des Neurons => Dämpfung auf 75% verkleiner          
                    pass
        return self.hs3
        pass

    
    # train dynNN

    # query dynNN
    
#####################################################################
# Schritt 0: Initialisieren von allen Arealen des DNN (Gewichte, Dämpfung,...  #
input_nodes = 783               # MNIST Datensatz 28x28=784 Pixel 
hidden_nodes = 199  
output_nodes = 9               # Detektion von 10 Ziffern aus dem MNIST Datensatz
stufen_wert = 1           # Stufenwert der Sprungfunktion
alpha_takt = 30             # Anzahl der Takte (=Schwingungsdauer); der Alpha-Welle = 30; Betha-Welle = 90Takte
alpha_step = 0              # der Startwert der Alpha-Welle
alpha_bias = 0            # Verschiebung der sin-Welle in den Plus-Bereich (1) + etwas höher (0.2)
input_times = 70            # Anzahl der Lerndurchläufe

# create instance of 3 dynneural network areals
n = dynNN1(input_nodes, hidden_nodes, output_nodes)
m = dynNN2(input_nodes, hidden_nodes, output_nodes)
o = dynNN3(input_nodes, hidden_nodes, output_nodes)
n.status1()
m.status2()
o.status3()

print("Anzahl input nodes= ", input_nodes+1)
print("Anzahl hidden nodes= ", hidden_nodes+1)
print("Anzahl output nodes= ", output_nodes+1)
print("alpha_takt= ", alpha_takt)
print("alpha_step= ", alpha_step)
print("alpha_bias= ", alpha_bias)
print("input_times= ", input_times)

# Vektor: hi_alt Vektor aufstellen (merken für den Vergleich mit aktuellem hi)
hi_alt1 = numpy.zeros( [hidden_nodes] )

# Vektor: hs und hs_alt Vektor aufstellen (speichert die jeweilige Sprungantwort enstpr. des stufen_wert)
n.hs1 = numpy.zeros( [hidden_nodes] )
n.hs_alt1 = numpy.zeros( [hidden_nodes] )

# Vektor: os Vektor aufstellen 
n.os1 = numpy.zeros( [output_nodes] )

# Vektor: C und B für das numpy array definieren
C1 = numpy.zeros( [output_nodes,input_times,] )
B1 = numpy.zeros( [output_nodes,input_times,] )
C2 = numpy.zeros( [output_nodes,input_times,] )
B2 = numpy.zeros( [output_nodes,input_times,] )
C3 = numpy.zeros( [output_nodes,input_times,] )
B3 = numpy.zeros( [output_nodes,input_times,] )

# Schritt 1: Schleife über Anlegen aller bekannten Inputsignale (Alphabet)
# (Anm. hier jetzt statisch angelegter Input-Vektor 
# Vektor: load Input Vektor: Inp[Spalte] ... hier noch manuelle Eingabe!
# i11
# i21    wobei ixy=> x = input nummer(i);  y = Zeitpunkt (t): 0, 1, 2, 3, etc.
Input = numpy.zeros( [input_nodes] ) + 0.01
Input[0] = 0.39
Input[1] = 0.99
Input[2] = 0.29
Input[3] = 0
Input[4] = 0
Input[5] = 0
Input[6] = 0.19
Input[16] = 0.29
Input[27] = 0.89
Input[38] = 0
Input[49] = 0
Input[50] = 0
Input[60] = 0.69
Input[701] = 0.79
Input[80] = 0.99
Input[90] = 0
Input[100] = 0
Input[110] = 0
Input[120] = 0.09
Input[160] = 0.19
Input[270] = 0.79
Input[380] = 0
Input[390] = 0
Input[400] = 0
# (Anm. hier zunächst nur 1,1,1,.. dann 0,0,0,.. abwechselnd 
# Vektor: load Input Vektor: Inp[Zeile,Spalte] ... hier noch manuelle Eingabe!
# i11  i12
# i21  i22    wobei ixy=> x = input nummer(i);  y = Zeitpunkt (t): 0, 1, 2, 3, etc.
#dynamisch angelegter Input 1,1,1,1 dann 0,0,0,0 dann wieder 1,1,1,1    
#Input = numpy.zeros( [input_nodes, input_times] )
#INPUT_akt = numpy.zeros( [input_nodes] )
#for z in range(input_times):
#    for i in range(input_nodes):
#        if z%2 == 0:
#            Input[i,z] = 1
#            #print("i=" "ist gerade")
#        else:
#            Input[i,z] = 0
#            #print("i=" "ist ungerade")
###print("Input= \n{}".format(Input))
#print(Input)   
print()

###############################################
######HAUPTPROGRAMM SCHLEIFE 1#################
###############################################

# Schritt 2: Schleife über alle Lernzyklen
for z in range(input_times): # input_times = Anzahl der Durchläufe
    ###print("------------------------------------------------")
    ###print("Zeitpunkt= ", z, "Stufen_wert= ", stufen_wert)
    # Schritt 3: Schleife über alle Buchstaben (Hier zunächst nur ein 0,0,0,0,0,0, oder 1,1,1,1,1,1 Muster
    INPUT_akt = Input
#    for i in range(input_nodes):
#        INPUT_akt[i] = Input[i,z]
    ###print("INPUT_akt= \n{}".format(INPUT_akt))
     
    # Schritt 4: Schleife über alle Areale (Hier zunächst nur ein Areal)

    # Schritt 5: Schleife über eine komplette Alpha-Welle (= Stufenwert modulieren)

    # Schritt 6: Vektor hi berechnen: hi = wih @ INPUT_akt[i]-Vektor + dihh @ self.hidden_outputs
    n.vektor_hi1(INPUT_akt)
    m.vektor_hi2(INPUT_akt)
    o.vektor_hi3(INPUT_akt)
    #print("hidden_outputs vor Sprung  = \n{}".format(n.hidden_outputs.round(3)))

    # Schritt 7: Sprungantwort der hidden Neuronen entspr. dem Stufenwer (stufen_wert) berechnen
    n.sprung_antwort_hidden1(stufen_wert)
    m.sprung_antwort_hidden2(stufen_wert)
    o.sprung_antwort_hidden3(stufen_wert)
    ###print("hidden_outputs nachher = \n{}".format(n.hidden_outputs.round(3)))

    # Schritt 8: dynamische Anpassung der Dämpfungsgewichte di
    # wenn ein Neuron immer 0,1,0,1,0,1,... (=Chaos) zeigt, damm Dämpfungen di von diesem Neuron aus erhöhen
    # wenn ein Neuron immer 0,0,0,0,0,0,... (=Tot) zeigt, damm Dämpfungen di von diesem Neuron aus verkleinern.
    
    # Schritt 9: Vektor ho berechnen: ho = who @ self.hidden_outputs-Vektor + dohh @ self.output_outputs
    n.vektor_ho1()
    m.vektor_ho2()
    o.vektor_ho3()
    
    #print("output_outputs vor Sprung = \n{}".format(n.output_outputs.round(3)))
    for i in range(output_nodes):
        B1[i,z]=n.output_outputs1[i]
        B2[i,z]=m.output_outputs2[i]
        B3[i,z]=o.output_outputs3[i]


    # Schritt 10: Sprungantwort der output Neuronen o1 - on entspr. dem Stufenwert berechnen
    n.sprung_antwort_output1(stufen_wert)
    m.sprung_antwort_output2(stufen_wert)
    o.sprung_antwort_output3(stufen_wert)
    #print("output_outputs nachher = \n{}".format(n.output_outputs.round(3)))
    ###print(format(n.output_outputs.round(3)))
    
    
    # Schritt 11: dynamische Anpassung der Dämpfungsgewichte do
  
    # Schritt 12: Alpha Modellierung des Stufenwertes
    # Refraktionszeit = 3ms; alpha-Welle ca 100ms (=10Hz); ß-Welle ca. 30Hz => Lernschritte zwischen 30 - 90 Takten (alpha - ß Welle)
    #print("z%alpha_takt= ", z%alpha_takt) # z%alpha_takt = z modulo alpha_takt => gibt immer nur den Rest der Division zurück
    stufen_wert = numpy.sin(2*math.pi/alpha_takt*(z%alpha_takt))+alpha_bias
    
    ###print("Zeitpunkt= ", z, "stufen_wert= ", stufen_wert) 
    #print("Lernschritt, Schritt", Lernschritt, Schritt)

    for i in range(output_nodes):
        C1[i,z]=n.output_outputs1[i]
        C2[i,z]=m.output_outputs2[i]
        C3[i,z]=o.output_outputs3[i]
            

#plt.imshow( B, interpolation="nearest")
#plt.yscale(3)
ax = plt.gca()
ax.set_ylim( [0,output_nodes] )
plt.subplot(221)
plt.imshow( C1, interpolation="nearest")
plt.subplot(222)
plt.imshow( C2, interpolation="nearest")
plt.subplot(223)
plt.imshow( C3, interpolation="nearest")
plt.show()

# Schritt 13: Synchronisation pro Buchstabe über alle Areale berechnen und wenn:
# a) Synchron.Ergebnis DNNneu besser als DNN alt dann DNNneu behalten sonst
# b) DNNalt behalten und die Gewichte der Verknüpfungen hi und oi per Zufall und die Dämpfungen di und do neu setzen (Gaußsche Normalvert.).

    









    
        
        
    
