Programmet kjøres i følgende rekkefølge for modellering:

Databehandling: Først kjøres datatilberedning.ipynb, som oppretter to CSV-filer. Én fil brukes til visualisering, og den andre til modellering. Dette gir fleksibilitet til å kjøre visualisering uavhengig av modelleringen – og motsatt.

Visualisering: Deretter kjøres visualiseringsnotatboken, som gir innsikt i dataene etter databehandlingens andre steg.

Modellering: Til slutt kjøres modelleringen, der modellen lagres slik at den kan benyttes videre på nettsiden.

Etter dette kan man starte nettsiden med å kjøre app.py, og gå videre på lokalhosten som kommer opp.

Alle notebooks kjøres linje for linje og trenger ikke noe annet rekkefølge enn dette.