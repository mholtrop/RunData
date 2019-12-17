import sys
import os
import re

def CompareLists(listA, listB):
    missinginA = []
    missinginB = []

    PresentinBoth = []

    for runA in listA:
        if runA in listB:
            PresentinBoth.append(runA)
        else:
            missinginB.append(runA)

    for runB in listB:
        if runB in listA:
            continue
        else:
            missinginA.append(runB)

    CompareDict = dict()
    CompareDict["missinginA"] = missinginA
    CompareDict["missinginB"] = missinginB
    CompareDict["PresentinBoth"] = PresentinBoth

    return CompareDict





MauriksList = []
CameronsList = []

with open("Runs2019_withCharge.dat") as ff:
    for line in ff:
        line = line.replace('\n', '')
        MauriksList.append(line)


with open("CameronsList.dat") as ff:
    for line in ff:
        line = line.replace('\n', '')
        CameronsList.append(line)


compareResult = CompareLists(MauriksList, CameronsList)

print (" =============================== Maurik's List ===================================== ")
print ("# of runs in Maurik's list is " + str(len(MauriksList) ) )
print (MauriksList)

print (" =============================== Cameron's List ===================================== ")
print ("# of runs in Cameron's list is " + str(len(CameronsList) ) )
print (CameronsList)


print ("=============================== Runs in both list ====================================")
print ("# of common runs in Bith list is " + str(len(compareResult["PresentinBoth"]) ) )
print (compareResult["PresentinBoth"])
for curRun in compareResult["PresentinBoth"]:
    print (curRun)

print ("=============================== Missing in Maurik's list ====================================")
print (compareResult["missinginA"])


print ("=============================== Missing in Cameron's list ====================================")
print (compareResult["missinginB"])

for curRun in compareResult["missinginB"]:
    print (curRun)
