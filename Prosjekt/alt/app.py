from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd 
import pickle

app = Flask(__name__)
# Funksjon som må brukes for bruk av pipeline
def convert_to_str(x):
    return x.astype(str)

# Last inn den forhåndstrente modellen
with open("model.pkl", "rb") as model_file:
    best_model_pipeline_fill_zero = pickle.load(model_file)

# Samme variabelutvinning som i datatilbredningen  
# Funksjon for å lage aldersgrupper
def kalk_alder_gruppe(alder):
    return pd.cut([alder], bins=[0, 18, 30, 45, 60, 75, np.inf], labels=[0, 1, 2, 3, 4, 5])[0]

# Funksjon for å mappe inntekt
inntekt_mapping = {
    "under $11k": 0,
    "$11-$25k": 1,
    "$25-$50k": 2,
    ">$50k": 3
}
def map_inntekt(inntekt):
    return inntekt_mapping.get(inntekt, np.nan)  # Setter til NaN hvis inntekt ikke finnes i mappingen

# Funksjon for å beregne sosioøkonomisk status
def kalk_sosiooekonomisk_status(inntekt, utdanning):
    if pd.notnull(inntekt) and pd.notnull(utdanning):
        return (inntekt + utdanning) / 2
    return np.nan

# Funksjon for overlevelsesestimat
def kalk_overlevelses_proxy(overlevelsesestimat_2mnd, overlevelsesestimat_6mnd, lege_overlevelsesestimat_2mnd,
                                 lege_overlevelsesestimat_6mnd, fysiologisk_score, apache_fysiologisk_score):
    return np.mean([overlevelsesestimat_2mnd, overlevelsesestimat_6mnd, lege_overlevelsesestimat_2mnd,
                    lege_overlevelsesestimat_6mnd, fysiologisk_score, apache_fysiologisk_score])

# Funksjon for nyrefunksjonsproxy
def kalk_nyrefunksjons_proxy(kreatinin, blodurea_nitrogen):
    return np.mean([kreatinin, blodurea_nitrogen])


# Funksjon for å konvertere og validere input
def validate_and_prepare_input(data):
    try:
        # Valider og konverter input som tidligere
        alder = float(data.get('alder'))
        kjønn = data.get('kjønn') 
        utdanning = float(data.get('utdanning'))
        inntekt = data.get('inntekt')
        inntekt_mapped = map_inntekt(inntekt)
        etnisitet = data.get('etnisitet')

        # Andre variabler som valideres som før
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
        adl_pasient = data.get('adl_pasient')
        dnr_dag = data.get('dnr_dag')

        # Bruk variabelutvinningene
        alder_gruppe = kalk_alder_gruppe(alder)
        sosiooekonomisk_status = kalk_sosiooekonomisk_status(inntekt_mapped, utdanning)
        overlevelses_proxy = kalk_overlevelses_proxy(
            overlevelsesestimat_2mnd, overlevelsesestimat_6mnd, lege_overlevelsesestimat_2mnd,
            lege_overlevelsesestimat_6mnd, fysiologisk_score, apache_fysiologisk_score
        )
        nyrefunksjons_proxy = kalk_nyrefunksjons_proxy(kreatinin, blodurea_nitrogen)

        # Lag dictionary med kolonner modellen forventer
        input_data = {
            'alder': alder,
            'kjønn': kjønn,
            'utdanning': utdanning,
            'inntekt': inntekt_mapped,
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
            'lege_overlevelsesestimat_6mnd': lege_overlevelsesestimat_6mnd,
            'alder_gruppe': alder_gruppe,
            'sosiooekonomisk_status': sosiooekonomisk_status,
            'overlevelses_proxy': overlevelses_proxy,
            'nyrefunksjons_proxy': nyrefunksjons_proxy,
            'adl_pasient': adl_pasient,
            'dnr_dag': dnr_dag 
        }

        # Konverter til DataFrame
        input_df = pd.DataFrame([input_data])
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

    print(prepared_data)
    
    if not valid:
        return jsonify({"error": prepared_data}), 400  # Returner feilmelding ved feil input

    # Gjør prediksjon med modellen
    prediction = best_model_pipeline_fill_zero.predict(prepared_data)  # Forventet en DataFrame
    prediction = np.round(prediction)
    # Send brukeren til result.html med prediksjonen
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
