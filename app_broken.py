# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle # do pobrania modelu uczenia maszynowego
from datetime import datetime
startTime = datetime.now() ## te 2 linijki standardowo do każdej appki streamlitemowej
# import znanych nam bibliotek

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath # wykorzystanie ścierzk windowsowej

filename = "model.sv"
model = pickle.load(open(filename,'rb'))# władowanie modelu do pamięcia aplikacji
# otwieramy wcześniej wytrenowany model

sex_d = {0:"Kobieta",1:"Mężczyzna"} ## dodanie nowej pł
pclass_d = {0:"Pierwsza",1:"Druga", 2:"Trzecia"}
embarked_d = {0:"Cherbourg", 1:"Queenstown", 2:"Southampton"}
# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem

def main():

	st.set_page_config(page_title="Heart Disease")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	st.image("https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.hMntgqk-9p2iOdXQWOtILgHaFP%26pid%3DApi&f=1&ipt=2c13cd14b89a7be9a174dd1de1431d70a62bc1cafa0da8f620621012bda1ccb8&ipo=images")

	with overview:
		st.title("Heart Disease")

	with left:
		objawy = st.slider("Objawy", value=1, min_value=0, max_value=9, step=1)
		age_slider = st.slider("Wiek", value=1, min_value=1, max_value=90, step=1)
		choroby_wsp = st.slider("Choroby współistniejące", value=0, min_value=0, max_value=8, step=1)
		wzrost = st.slider("Wzrost", value=160, min_value=120, max_value=230, step=1)
		leki = st.slider("Liczba leków", value=1, min_value=1, max_value=8, step=1)
	#data = [[pclass_radio, sex_radio,  age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio]]
	data = [[objawy, age_slider, choroby_wsp, wzrost, leki]]
	survival = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.subheader("Czy taka osoba jest zdrowa?")
		st.subheader(("Tak" if survival[0] == 1 else "Nie"))
		st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()
