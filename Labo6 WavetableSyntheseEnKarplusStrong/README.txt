Labo 6: Wavetable-synthese en het Karplus-Strong algoritme

Auteur: Lorin Speybrouck
Datum: 8/11/2024

--------------------------------------------------------------------------------

OS: Windows 11

Versie controle:
    Python 3.13.3
    Opencv 4.10.0
    Numpy 2.1.1
    Scipy 1.14.1
    Matplotlib 3.9.2

IDE: Visual Studio Code

--------------------------------------------------------------------------------

Bestanden/Mappen:
images/
    map met input afbeeldingen

labo6.py
    python script dat de oefeningen van labo6 sequentieel uitvoert

start commando: python labo6.py 

--------------------------------------------------------------------------------

Vragen:
1. Vraag: Wat is het gevolg van het feit dat p een geheel getal moet zijn?
Antwoord: 
- De toonhoogte kan licht afwijken van de gewenste frequentie. Omdat p afgerond wordt, kan dit resulteren in een frequentie die iets lager of hoger is dan de gewenste frequentie

2. Vraag: Wat is de frequentie voor volgende noten: B4, F5, G#6 en D7?
Antwoord: 
- B4:  493.88 Hz
- F5:  698.452 Hz
- G#6: 1661.209 Hz
- D7:  2349.304 Hz

3. Vraag: Welk effect heeft de frequentie op het verloop van de geluidsgolf ? Wat zie je nog op het frequentiespectrum?
Antwoord:
- Een hogere frequentie lijd tot een kortere toon
- In het frequentiespectrum zie je voor elke noot een sterke piek bij de fundamentele frequentie en daarnaast harmonischen die bijdragen aan de timbre van het geluid

4. Vraag: Met welke verhouding komt deze uitrekkingsfactor overeen?
Antwoord:
- De uitrekkingsfactor S=2^(octaaf−1) komt overeen met een verdubbeling van de vertragingsbufferlengte voor elke verhoging van een octaaf.