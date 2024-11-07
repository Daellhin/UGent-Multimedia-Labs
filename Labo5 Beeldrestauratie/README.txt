Labo 5: Beeldrestauratie

Auteur: Lorin Speybrouck
Datum: 25/10/2024

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

labo5.py
    python script dat de oefeningen van labo4 sequentieel uitvoert

start commando: python labo5.py 

--------------------------------------------------------------------------------

Vragen:
1. Een eerste reden hiervoor is numerieke stabiliteit Waarom? Wat is het gevaar bij een deling? 
Hint: Denk aan wat er gebeurt bij frequentiecomponenten met een lage magnitude.
Antwoord:
- Numerieke onstabiliteit ontstaat omdat computers getallen maar met een beperkte precisie kunnen voorstellen(afhankelijk van het aantal bits)
- Zeer kleine komma getallen(opgeslagen als floating point) hebben dus maar een beperkt aantal getallen na de nul, en bij deling zal dit waarschijnlijk afgerond worden

2. Voor welke waarde van k is de werking van een Wiener filter dezelfde als die van een invers filter?
Antwoord: 
- Bij k=0

3. In een meer geavanceerde vorm van het Wiener filter kan de constante k worden vervangen door een term die afhankelijk is van de frequenties (u en v). Wat is het voordeel hiervan (denk aan de verschillende soorten ruis)? Waarom is dit moeilijk toepasbaar in de praktijk?
Antwoord: 
- Ruis is vaak frequentie afhankelijk, bij sommige frequenties meer ruis dan anderen, wat zeker een voordeel kan zijn
- Maar het is moeilijk om exact te bepalen hoeveel ruis er op een frequentie is, wat het moeilijk toepasbaar maakt

4. Vraag: Met welk concept uit de signaalverwerking komt deze waarde gemiddeld gezien overeen, wanneer ze goed gekozen is voor een bepaald beeld? 
Hint: we zoeken een specifieke verhouding.
Antwoord: 
- Het aanpassen van de k-waarde in het Wiener filter is gerelateerd aan de Signal-to-Noise Ratio(SNR). 
- Wanneer de k-waarde optimaal gekozen wordt voor een bepaald beeld, komt deze waarde meestal overeen met 1/SNR
