from flask import Flask, request, render_template
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pandas as pd

app = Flask(__name__)

# Laster datasettet
df = pd.read_csv('Lab-7\\titanic.csv')

# Encoderer kategoriske variabler
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# Trener modellen
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']
X = X.dropna()
y = y[X.index]
model = LogisticRegression()
model.fit(X, y)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Henter inn nye data fra formen
        pclass = int(request.form['pclass'])
        sex = int(request.form['sex'])
        age = int(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        fare = float(request.form['fare'])

        # Lager en ny DataFrame med nye data
        new_data = pd.DataFrame({'Pclass': [pclass], 'Sex': [sex], 'Age': [age], 'SibSp': [sibsp], 'Parch': [parch], 'Fare': [fare]})

        # Bruker modellen til å gjøre en prediksjon
        prediksjon = model.predict(new_data)

        # prediksjonen: 1 = død, 0 = overlevende
        if prediksjon[0] == 1:
            resultat = "Død"
        else:
            resultat = "Overlevende"

        # Returnerer prediksjonen
        return render_template('resultat.html', prediksjon=resultat)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8080)