#!/usr/bin/python3
#----------------------------------------------------------------
# Mein Dynamisches Neuronales Netz
# Dateiname: Mein_DNN_1018.py
# R.J.Nickerl mit github
# 05.04.20 Python 3.8
#--------------------------------------------------------------
try:
    import tkinter as tk
except ImportError:
    import Tkinter as tk                        
from random import randint
import numpy
import scipy.special
import matplotlib.pyplot
#import math
#import time
#import RPi.GPIO as GPIO
#import datetime

# Dynamisches Neuronales Netz class definition
class dynNN:

    # initialise the dynNN
    def __init__(self,inputnodes, hiddennodes, outputnodes, learningrate, stufenwert, maxverzoegerung):
        # set number of nodes in each input, hidden, output and daempfungs layer
        self.inodes = inputnodes        # Anzahl Input Nodes
        self.hnodes = hiddennodes       # Anzahl Hidden Nodes
        self.dnodes = self.hnodes       # Anzahl Damp Nodes (=Dämpfungsknoten)
        self.onodes = outputnodes       # Anzahl Output Nodes
        self.lr     = learningrate      # Anzahl Lerndurchläufe        
        self.swert  = stufenwert        # Höhe des Schwellwertes der Hidden Nodes bei dem der Knoten "schaltet"
        self.mverz  = maxverzoegerung   # Anzahl der maximalen Verzögerungsläufe bevor ein Hidden Neuron dämpft
        
        # link weight matrices: wih, who and dhh
        # weights inside the arrays are:
        # w_i_j, where link is from node i to node j in the next layer
        # d_i_j, where reverse link is from node i to node j in the same layer
        # w11 w21   d11 d21
        # w12 w22   d12 d22
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.dhh = numpy.random.normal(0.0, pow(self.dnodes, -0.5), (self.dnodes, self.dnodes))
        pass
    
        # Vector: D[i] Dämpfungsnode werden random erstbesetzt
        # D[1]
        # D[2] etc.
        self.D_start = numpy.random.randint(self.mverz, size=(self.dnodes))
        self.D_akt = numpy.zeros( [self.dnodes] )
        for i in range(self.dnodes):
            self.D_akt[i] = self.D_start[i]
        print("D_start= ", self.D_start)
        print("D_akt= ", self.D_akt)

        
        # acivation function is the sigmoid function Änderung: hier wird die Stufenfunktion verwendet
        self.activation_function = lambda x: scipy.special.expit(x)

    # status dynNN
    def status(self):
        print("Anzahl input nodes= ", self.inodes)
        print("Anzahl hidden nodes= ", self.hnodes)
        print("Anzahl output nodes= ", self.onodes)
        print("Anzahl damp nodes= ", self.dnodes)
        print("wih= ")
        print(self.wih.round(3))
        print("who= ")
        print(self.who.round(3))
        print("dhh= ")
        print(self.dhh.round(3))
        #print()         
        pass

    # search dynNN
    def search(self, inputs_list):
        inputs = inputs_list
        print("SUCHE...............")
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        print(self.wih, inputs)
        pass

    # Vektor hi berechnen
    def vektor_hi(self, inputs_list):
        print("cccc vektor_hi cccc")
        inputs = inputs_list
        self.hidden_inputs = numpy.dot(self.wih, inputs)
        print("xxxxxclass hidden_inputs= \n{}".format(self.hidden_inputs.round(3)))
        return self.hidden_inputs
        pass
    
    # Sprungantwort hs Vektor berechnen
    def sprung_antwort(self):
        print("cccc sprung_antwort cccc")
        #print("hs vorher= \n{}".format(self.hs))
        for i in range(self.hnodes):
            self.hs_alt[i] = self.hs[i]             # vorher den alten "hs" als "hi_alt" abspeichern
            #print("Spriúng D_akt vorher = ", self.D_akt[i])
            #D_akt[i] = D_akt[i] -1
            #print("Spriúng D_akt nachher= ", D_akt[i])
            #print("self.hidden_inputs[i]xxx= ", self.hidden_inputs[i], i)
            if self.hidden_inputs[i] > self.swert:
                self.hs[i] = 1
            else:
                self.hs[i] = 0
        print("hs alt= \n{}".format(self.hs_alt))
        print("hs NEU= \n{}".format(self.hs))
        #print("self.hidden_inputs[i]xxx= ", self.hidden_inputs[i], i)
        return self.hidden_inputs
        return self.hidden_inputs
        return self.hs
        return self.hs_alt
        pass
    
    # Ziel ist ein chaotisches Schwingen (0,1,0,1,0,1,...) zu detektieren und zum Zeitpkt.=0 zu dämpfen
    # Dämpfungsnodes um 1 = einen Zeitpunkt verkleinern
    # Dämpfungswerte dhh anpassen. 
    # Dämpfungsmatrix erzeugen wenn Zeitwert z=0 dann dämpfen
    def daempf_anpassung(self):
        print("cccc daempfung_anpassung cccc")
        #print("hs vorher= \n{}".format(self.hs))
        print("dhh vorher= \n{}".format(self.dhh.round(2)))        
        for i in range(hidden_nodes):
            for k in range(hidden_nodes):
                if self.hs_alt[i] != self.hs[i]:
                    self.dhh[i,k] = 1.25*self.dhh[i,k]  # Detektion von 0,1,0,.. Feuern des Neurons hs[i]=>Chaos=>Erhöhung der Dämpfung          
                    #print("hsalt ungl hsneu")
                    #print("self.hidden_inputs[i]= ", self.hidden_inputs[i], i)
                    #print("self.dhh[i,k]= ", self.dhh[i,k], i, k)
                    pass
                if self.hs_alt[i] == self.hs[i]:
                    self.dhh[i,k] = 0.75*self.dhh[i,k]  # Detektion von 0,0,0, oder 1,1,1.. kein/immer Feuern des Neurons => Dämpfung auf 75% verkleiner          
                    pass
                
        #print("hs nachher= \n{}".format(self.hs))
        print("dhh nachher= \n{}".format(self.dhh.round(2)))
        return self.hs
        pass

    
    # train dynNN

    # query dynNN
####################################################
# number of input, hidden, output and daempfungs nodes
input_nodes = 4
hidden_nodes = 5
output_nodes = 2

input_times = 4         # Anzahl der Lerndurchläufe
learning_rate = 0.1     # learning rate
stufen_wert = 0.5       # Stufenwert der Stufenfunktion
max_verzoeg = 5         # Maximale Verzögerungszeit der D-Neuronen

# create instance of dynneural network
n = dynNN(input_nodes, hidden_nodes, output_nodes, learning_rate, stufen_wert, max_verzoeg)
n.status()

# Vektor: load Input Vektor: Inp[Zeile,Spalte] ... hier noch manuelle Eingabe!
# i11  i12
# i21  i22    wobei ixy=> x = input nummer(i);  y = Zeitpunkt (t): 0, 1, 2, 3, etc.
Input = numpy.zeros( [input_nodes, input_times] )
INPUT_akt = numpy.zeros( [input_nodes] )
for t in range(input_times):
    for i in range(input_nodes):
        if t%2 == 0:
            Input[i,t] = 1
            #print("i=" "ist gerade")
        else:
            Input[i,t] = 0
            #print("i=" "ist ungerade")
print("Input= ")
print(Input)   
print()

# Vektor: hi_alt Vektor aufstellen (merken für den Vergleich mit aktuellem hi)
hi_alt = numpy.zeros( [hidden_nodes] )

# Vektor: hs und hs_alt Vektor aufstellen (speichert die jeweilige Sprungantwort enstpr. des stufen_wert)
n.hs = numpy.zeros( [hidden_nodes] )
n.hs_alt = numpy.zeros( [hidden_nodes] )

###############################################
######HAUPTPROGRAMM#######HAUPTPROGRAMM########
###############################################

for z in range(input_times): # input_times = Anzahl der Durchläufe
    print("------------------------------------------------")
    print("Zeitpunkt= ", z)
    # 0. Schritt Input zum Zeitpunkt z auslesen
    for i in range(input_nodes):
        INPUT_akt[i] = Input[i,z]
    print("INPUT_akt= ")
    print(INPUT_akt)
    
    # 1. Schritt Vektor hi berechnen: hi = wih @ INPUT_akt-Vektor
    n.vektor_hi(INPUT_akt)
    #print("hidden_inputs 2x = ")
    #print(n.hidden_inputs.round(1))

    # 2. Schritt Sprungantwort der hidden Neuronen hs (s=Sprung) entspr. dem Stufenwer (stufen_wert) berechnen
    n.sprung_antwort()

    # 3. Schritt Die Dämpfungsvektoren Di werden entsprechend der aktuellen Sprungantwort geändert:
    #            wenn hs = hi pro Zeitpunkt z um eins verkleinert, bis
    #            der Dämpfungswert = 0 ist, dann Hi berechnen: Hi = dhh @ ektor hi berechnen: hi = wih @ Inp-Vektor
    n.daempf_anpassung()
    print(n.hidden_inputs.round(1))
    print("hs NEU= \n{}".format(n.hs))
    print("hs_alt yyyyyyy= \n{}".format(n.hs_alt))
    #print("hs neu= ")
    #print(n.hs.round(1))    
    #print("hs_alt neu= ")
    #print(n.hs_alt.round(1))
            

# 4. Schritt Vektor hi berechnen: hi = wih @ Inp-Vektor









    
        
        
    
