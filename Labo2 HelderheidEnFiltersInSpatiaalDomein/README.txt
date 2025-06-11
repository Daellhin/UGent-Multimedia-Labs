Labo 2: Helderheidtransformaties en Filtering in het spatiale domein

Auteur: Lorin Speybrouck
Datum: 4/09/2024

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

labo2.py
    python script dat de oefeningen van labo2 sequentieel uitvoert

start commando: python labo2.py 

--------------------------------------------------------------------------------

Vragen:
1. Voor welke waarden van γ vergroot je het contrast in donkere zones, respectievelijk lichte zones?
Antwoord:
- γ kleiner dan 1: maakt donkere zones lichter, verhoogt het contrast in de donkere zones(verlaagt in lichte zones)
- γ groter dan 1:  maakt lichte zones donkerder, verlaagt het contrast in de donkere zones(verhoogt in lichte zones)

2. Geef aan wat de nadelen van deze methode zijn en wat de invloed van de grootte van het filter is. Bespreek ook waarom de som van alle elementen gelijk moet zijn aan 1.
Antwoord:
- Averaging filter verwijderd zowel ruis als beelddetails, alles word vager
- Werkt niet goed voor salt en peper noise
- Kleiner filters verminderen ruis en laten beeld relatief intact
- Grotere filters vermideren meer ruis maar vervagen de afbeeldingen meerµ

3. Vermeld wat de voordelen en nadelen zijn ten opzichte van een averaging filter. Let opnieuw op de grootte van het filter.
Antwoord
- De Gaussiaanse filter kent meer gewicht toe aan de centrale pixels, waardoor details in het beeld beter behouden blijven dan bij een averaging filter. Dit leidt tot minder vervaging van randen.
- De Gaussiaanse verdeling zorgt voor een gelijkelijke overgang van centraale naar randpixels, wat resulteert in een realistischer vervaging dan bij een averaging filter
- Bij zeer grote standaarddeviaties kunnen bij de Gaussiaanse filter de kleine details nog steeds verdwijnen, omdat de vervaging over een grotere afstand wordt verspreid.