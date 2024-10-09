Labo 1: Inleiding tot Python en basisbewerkingen met beelden

Auteur: Lorin Speybrouck
Datum: 3/10/2024
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

labo1.py
    python script dat de oefeningen van labo1 sequentieel uitvoert

start commando: python labo1.py 

--------------------------------------------------------------------------------
Vragen:
1. In welke volgorde worden de kleurkanalen ingelezen bij de functie cv2.imread en bij de functie plt.imread? Wat gebeurt er indien je de afbeelding inleest met OpenCV en daarna afbeeldt met pyplot
Antwoord: 
   - Bij cv2.imread worden de kleuren ingelezen in GRB formaat
   - Bij plt.imread worden de kleuren ingelezen in RGB formaat
   - De kleuren van de afbeelding worden fout weergegeven(blauwe huid)

2. Beschrijf in enkele regels, hoe je automatisch de hoge intensiteiten van de lagere zou kunnen scheiden voor dit soort röntgenbeelden aan de hand van het histogram. Hoe kies je hierbij automatisch de drempel-waarde? Beschrijf grondig hoe je dit aanpakt voor dit voorbeeld.
Antwoord: 
    - Door een drempelwaarde te gebruiken kunnen de hoge intensiteiten van de lagere gescheiden worden, alle intensiteiten hierboven worden dan wit(1) en alle intensiteiten er onder zwart(0) 
    - Met een histogram is er snel te zien welke pixel intensiteiten er het meest voorkomen. Het middelpunt kan dan gezocht worden dat zorgt dat het aantal intensiteitswaarden links en rechts gelijk zijn, dit maximaliseert de variantie tussen de twee intensiteitsklassen.

3. Waarom geeft deze functie het verwachte resultaat niet weer? Waarmee moet je rekening houden bij bewerkingen op afbeeldingen?
Antwoord:
    - De inputs zijn uint8, dit zijn unsigned dat types waardoor deze bij een aftrekking overrollen naar het maximum. Dit probleem is opgelost door te casten naar een signed data type, of het absolute verschil met cv2.absdiff te nemen
    - Er moet rekening gehouden worden met de data types van de afbeelding
