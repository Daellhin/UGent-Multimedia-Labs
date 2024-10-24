Labo 4: Filters in het frequentiedomein

Auteur: Lorin Speybrouck
Datum: 18/10/2024

--------------------------------------------------------------------------------

OS: Windows 11

Versie controle:
    Python 11.9
    Opencv 4.10.0
    Numpy 2.1.1
    Scipy 1.14.1
    Matplotlib 3.9.2

IDE: Visual Studio Code

--------------------------------------------------------------------------------

Bestanden/Mappen:
images/
    map met input afbeeldingen

labo4.py
    python script dat de oefeningen van labo4 sequentieel uitvoert

start commando: python labo4.py 

--------------------------------------------------------------------------------

Vragen:
1. Waarom geeft Sobel een betere ruisonderdrukking dan Prewitt?
Antwoord: Omdat het de randen van de kernel minder gewicht geeft, wat helpt om de invloed van ruis te verminderen


2. Wat is de minimale waarde van de s-parameter als je het RANSAC-algoritme zou gebruiken om cirkels in een beeld te detecteren?
Antwoord:
- Om een cirkel te definiëren, zijn er drie onafhankelijke punten nodig die niet op een lijn liggen. Deze drie punten zijn voldoende om de middelpunt- en straalparameters van de cirkel te berekenen.
- Dus de minimale waarde van de s-parameter voor cirkeldetectie met RANSAC is s=3
