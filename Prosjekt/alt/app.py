from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd 
import pickle

app = Flask(__name__)
# Funksjon som må brukes for bruk av pipeline
def convert_to_str(x):
    return x.astype(str)

# Last inn den forhåndstrente modellen
with open("Prosjekt\\alt\\model.pkl", "rb") as model_file:
    best_model_pipeline_fill_zero = pickle.load(model_file)

# Funksjon for å konvertere og validere input
def validate_and_prepare_input(data):
    try:
        # Sjekk og konverter til riktig datatype
        alder = float(data.get('alder'))
        kjønn = data.get('kjønn') 
        utdanning = float(data.get('utdanning'))
        inntekt = data.get('inntekt')
        etnisitet = data.get('etnisitet')  # Valider valg mellom "white", "black", "asian", "other"
        
        if etnisitet not in ['white', 'black', 'asian', 'other']:
            raise ValueError('Ugyldig verdi for etnisitet')

        sykehusdød = int(data.get('sykehusdød'))
        blodtrykk = float(data.get('blodtrykk'))
        hvite_blodlegemer = float(data.get('hvite_blodlegemer'))
        hjertefrekvens = float(data.get('hjertefrekvens'))
        respirasjonsfrekvens = float(data.get('respirasjonsfrekvens'))
        kroppstemperatur = float(data.get('kroppstemperatur'))
        lungefunksjon = float(data.get('lungefunksjon'))
        serumalbumin = float(data.get('serumalbumin'))
        bilirubin = float(data.get('bilirubin'))
        kreatinin = float(data.get('kreatinin'))
        natrium = float(data.get('natrium'))
        blod_ph = float(data.get('blod_ph'))
        glukose = float(data.get('glukose'))
        blodurea_nitrogen = float(data.get('blodurea_nitrogen'))
        urinmengde = float(data.get('urinmengde'))
        sykdomskategori_id = data.get('sykdomskategori_id')
        sykdomskategori = data.get('sykdomskategori')
        dødsfall = int(data.get('dødsfall'))
        sykdom_underkategori = data.get('sykdom_underkategori')
        antall_komorbiditeter = int(data.get('antall_komorbiditeter'))
        koma_score = float(data.get('koma_score'))
        adl_stedfortreder = float(data.get('adl_stedfortreder'))
        fysiologisk_score = float(data.get('fysiologisk_score'))
        apache_fysiologisk_score = float(data.get('apache_fysiologisk_score'))
        overlevelsesestimat_2mnd = float(data.get('overlevelsesestimat_2mnd'))
        overlevelsesestimat_6mnd = float(data.get('overlevelsesestimat_6mnd'))
        diabetes = int(data.get('diabetes'))
        demens = int(data.get('demens'))
        kreft = data.get('kreft')
        lege_overlevelsesestimat_2mnd = float(data.get('lege_overlevelsesestimat_2mnd'))
        lege_overlevelsesestimat_6mnd = float(data.get('lege_overlevelsesestimat_6mnd'))

        # Lag en dictionary med kolonnenavnene modellen forventer
        input_data = {
            'alder': alder,
            'kjønn': kjønn,
            'utdanning': utdanning,
            'inntekt': inntekt,
            'etnisitet': etnisitet,
            'sykehusdød': sykehusdød,
            'blodtrykk': blodtrykk,
            'hvite_blodlegemer': hvite_blodlegemer,
            'hjertefrekvens': hjertefrekvens,
            'respirasjonsfrekvens': respirasjonsfrekvens,
            'kroppstemperatur': kroppstemperatur,
            'lungefunksjon': lungefunksjon,
            'serumalbumin': serumalbumin,
            'bilirubin': bilirubin,
            'kreatinin': kreatinin,
            'natrium': natrium,
            'blod_ph': blod_ph,
            'glukose': glukose,
            'blodurea_nitrogen': blodurea_nitrogen,
            'urinmengde': urinmengde,
            'sykdomskategori_id': sykdomskategori_id,
            'sykdomskategori': sykdomskategori,
            'dødsfall': dødsfall,
            'sykdom_underkategori': sykdom_underkategori,
            'antall_komorbiditeter': antall_komorbiditeter,
            'koma_score': koma_score,
            'adl_stedfortreder': adl_stedfortreder,
            'fysiologisk_score': fysiologisk_score,
            'apache_fysiologisk_score': apache_fysiologisk_score,
            'overlevelsesestimat_2mnd': overlevelsesestimat_2mnd,
            'overlevelsesestimat_6mnd': overlevelsesestimat_6mnd,
            'diabetes': diabetes,
            'demens': demens,
            'kreft': kreft,
            'lege_overlevelsesestimat_2mnd': lege_overlevelsesestimat_2mnd,
            'lege_overlevelsesestimat_6mnd': lege_overlevelsesestimat_6mnd
        }

        # Konverter input_data til en DataFrame
        input_df = pd.DataFrame([input_data])  # Modellen forventer en DataFrame med disse kolonnene
        return input_df, True

    except ValueError as e:
        return str(e), False

# Rute for startsiden
@app.route('/')
def index():
    return render_template('index.html')  # Dette vil laste HTML-skjemaet

@app.route('/predict', methods=['POST'])
def predict():
    # Hent data fra skjemaet
    data = request.form

    # Valider og konverter input
    prepared_data, valid = validate_and_prepare_input(data)

    if not valid:
        return jsonify({"error": prepared_data}), 400  # Returner feilmelding ved feil input

    # Gjør prediksjon med modellen
    prediction = best_model_pipeline_fill_zero.predict(prepared_data)  # Forventet en DataFrame
    prediction = np.round(prediction)
    # Send brukeren til result.html med prediksjonen
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
