Labo 3: Filters in het frequentiedomein

Auteur: Lorin Speybrouck
Datum: 11/10/2024

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

labo3.py
    python script dat de oefeningen van labo3 sequentieel uitvoert

start commando: python labo3.py 

--------------------------------------------------------------------------------

Vragen:
1. Volgens welk patroon komen deze nulpunten voor in het spectrum? En waarom?
Antwoord:
- Het patroon dat weerkeert is een rechthoek, dit komt omdat de oorspronkelijke figuur een rechthoek is en bij de randen(plotselinge overgangen) is de frequentie amplitude het hoogst. Dit herhaalt zicht in veelvouden van de rechthoek, zoals harmonischen
- Dit valt ook te vergelijken met het nemen van de Fourier getransformeerde  van een blokgolf

2. Waarmee komt de frequentie-inhoud met hoogste magnitude overeen?
Antwoord:
- De frequentie-inhoud met de hoogste magnitude, in dit geval een horizontale en verticale lijn door de oorsprong komt respectievelijk overeen met de verticale en horizontale lijnen in de afbeelding.

3. Verklaar wat er gebeurd als je de straal van de cirkelvormig filter laat variëren in grootte.
Antwoord:
- Hoe kleiner de straal van de afbeelding word hoe slechter de kwaliteit word
- Rond 125 is er al duidelijke ringing zichtbaar
- Boven 200 is er niet veel kwaliteitsverlies

4. Voer dit uit en bespreek de bovenstaande effecten.
Antwoord:
- Het ringing-effect ontstaat doordat het ideale laagdoorlaatfilter in het frequentiedomein een scherpe discontinuïteit heeft, wat resulteert in oscillaties (ringing) in het spatiale domein, veroorzaakt door de sinc-functie als inverse van het filter. Dit wordt versterkt door het Gibbs-fenomeen, dat optreedt bij abrupte overgangen, en leidt tot ongewenste variaties in intensiteit nabij randen. Het minimaliseren van deze effecten is cruciaal bij filterontwerp.
- Bij het transformeren van de ideale filter naar het spatiale domein is er ook ringing zichtbaar

5. Bespreek deze voordelen in commentaar, maar ook de nadelen.
Antwoord:
- Voordelen
    - Zachte overgang: Onderdrukt geleidelijk hoge frequenties, wat zorgt voor vloeiende vervaging zonder scherpe grenzen.
    - Ruisonderdrukking: Vermindert effectief hoge-frequentieruis zonder het beeld drastisch te verstoren.
    - Isotrope werking: Vervaagt gelijkmatig in alle richtingen, wat leidt tot consistente resultaten.
    - Weinig artefacten: Minder kans op "ringing" en aliasing-artefacten in vergelijking met harde filters.
- Nadelen
    - Detailverlies: Kan fijne beeldkenmerken en randen vervagen, wat ongewenst kan zijn in gedetailleerde beelden.
    - Slechte randbehoud: Vervaagt ook randen van objecten, wat leidt tot verlies van scherpte.

6. Bespreek wat er gebeurd als D0 varieert.
Antwoord:
- D0 bepaalt de grootte van de Gauss curve, en dus de grootte van de filter
- Een kleinere D0 resulteert in een kleinere filter en dus enkel de lage frequenties die doorgelaten worden, dit resulteert in een wazig beeld

6. Enkele nadelen van het Gaussiaans filter worden opgevangen in een 2de orde Butterworth laagdoorlaatfilter. Bespreek welke.
Antwoord:
- Scherpere overgangsband: Butterworth heeft een scherpere scheiding tussen de doorlaat- en stopband, waardoor ongewenste hoge frequenties beter worden onderdrukt zonder een te geleidelijke overgang.
- Minder vervaging: Het Butterworth-filter behoudt meer details en zorgt voor minder vervaging langs randen in vergelijking met het Gausiaanse filter.
