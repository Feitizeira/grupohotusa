import streamlit as st
import pandas as pd
import pickle
from config import PAGE_CONFIG

st.set_page_config(**PAGE_CONFIG)

def main():

    st.image("data/logo.png")

    tab0, tab1 = st.tabs(["Intro", "Modelo"])

    with tab0:
        
        st.write("Data Scientist Test con **:blue[streamlit]** y **:blue[sklearn]**.")

        st.markdown("""Los datos de este proyecto han sido proporcionados por Grupo Hotusa.""")

        st.write("""El modelo en producción es una red neuronal **:blue[MLPClassifier]** de la librería sklearn.""")

        st.info("""Nota: toda la información proporcionada por esta app es orientativa y basada en los datos proporcionados.""")
        
        st.write("""**Datos**""")

        dfo = pd.read_csv("data/hotusa_cancellations.csv")
        st.dataframe(dfo)
        df = pd.read_csv("data/hotusa_EDA.csv")
        
 
    with tab1:

        st.write("Predicción de Cancelación de las Reservas")
        st.caption("Neural Network MLPClassifier de sklearn. Accuracy del 96%. Recall 88%. Precision 97%")
        st.image("data/clasreportMLPb.png")
                 
        X = df.drop(['IsCanceled'], axis=1)
 
        feature_names = X.columns

        with st.form("Formulario:"):

                col1, col2, col3  = st.columns(3)
     
                with col1:
                        LeadTime = st.number_input(label = "Días de antelación de la reserva", 
                                                    # placeholder = "introduce el número",
                                                    value = 30
                                )
                with col2:
                        StaysInWeekendNights = st.number_input(label= "Noches en fin de semana de la reserva",
                                                # placeholder = "introduce el número de noches",
                                                value = 2
                                )
                with col3:
                        StaysInWeekNights = st.number_input(label= "Noches durante la semana",
                                                # placeholder = "introduce el número de noches",
                                                value = 0
                                )
                with col1:
                        Adults = st.number_input(label= "Número de adultos",
                                                # placeholder = "introduce el número de adultos",
                                                value = 2
                                )
                with col2:
                        Children = st.number_input(label= "Número de niños",
                                                # placeholder = "introduce el número de niños",
                                                value = 0
                                )
                with col3:
                        ADR = st.number_input(label= "Precio medio de la reserva por día de estancia",
                                                # placeholder = "introduce el precio medio",
                                                value = 92
                                )
                with col1:
                        Country = st.number_input(label= "Código del país",
                                                # placeholder = "introduce el código del país",
                                                value = 0
                                )
                with col2:
                        Company = st.number_input(label = "Código de la compañia", 
                                                # placeholder = "introduce el código",
                                                value = 0
                                )   
                with col3:
                        IsRepeatedGuest = st.radio(label= "Es un cliente que repite",
                                        options = ("Si", "No"), 
                                        index = 1,
                                        disabled = False,
                                        horizontal = True,
                                )
                with col1:
                        Mes_Estancia = st.slider(label= "Mes de la Estancia",
                                                min_value = 1,
                                                max_value= 12,
                                                step=1
                                )
                with col2:
                        Mes_Reserva = st.slider(label= "Mes de la Reserva",
                                                min_value = 1,
                                                max_value= 12,
                                                step=1
                                )
                with  col3:
                        ReservedRoomType = st.slider(label= "Tipo de Habitación reservada",
                                                min_value = 0,
                                                max_value= 8,
                                                step=1
                                )
                with col1:
                        Group = st.radio(label= "Reserva de Grupo",
                                        options = ("Si", "No"), 
                                        index = 1,
                                        disabled = False,
                                        horizontal = True,
                                )
                with col2:
                        Transient = st.radio(label= "En tránsito",
                                        options = ("Si", "No"), 
                                        index = 0,
                                        disabled = False,
                                        horizontal = True,
                                )
                with col3:
                        Transient_Party = st.radio(label = "En tránsito de fiesta", 
                                        options = ("Si", "No"), 
                                        index = 1,
                                        disabled = False,
                                        horizontal = True,
                                )         
                with col1:
                        SC = st.radio(label= "Tipo de alojamiento: Self-Catering",
                                        options = ("Si", "No"), 
                                        index = 1,
                                        disabled = False,
                                        horizontal = True
                                        )
                with col2:
                        HB = st.radio(label= "Tipo de alojamiento: Media Pensión",
                                        options = ("Si", "No"), 
                                        index = 1,
                                        disabled = False,
                                        horizontal = True,
                                )
                with col3:
                        FB = st.radio(label= "Tipo de alojamiento: Pensión Completa",
                                        options = ("Si", "No"), 
                                        index = 1,
                                        disabled = False,
                                        horizontal = True,
                                )            
                with col1:
                        BB = st.radio(label = "Tipo de alojamiento: Alojamiento y Desayuno", 
                                        options = ("Si", "No"), 
                                        index = 0,
                                        disabled = False,
                                        horizontal = True,
                                ) 
                with col2:
                        Undefined = st.radio(label = "Tipo de alojamiento: Sin determinar",
                                        options = ("Si", "No"), 
                                        index = 1,
                                        disabled = False,
                                        horizontal = True,
                                ) 
                with col3:
                        Contract = st.radio(label= "Contract",
                                        options = ("Si", "No"),  
                                        index = 1,
                                        disabled = False,
                                        horizontal = True,
                                ) 
              

                if st.form_submit_button("Enviar"):
                    datos = [LeadTime, StaysInWeekendNights, StaysInWeekNights, Adults,Children, ADR, 
                            Country, Company, ReservedRoomType,IsRepeatedGuest, Mes_Estancia, Mes_Reserva,
                            Contract, Group, Transient, Transient_Party, BB, FB, HB, SC, Undefined]

                    for i in range(len(datos)):

                        if datos[i] == 'Si':
                            datos[i] = 1
                                        
                        elif datos[i] == 'No':
                            datos[i] = 0

                #     dfdatos = pd.DataFrame(data = [datos], columns = feature_names)
                #     resultado = list(dfdatos.iloc[0].to_dict().values())
                   
                    df_usuario = pd.DataFrame(data = [datos], columns = feature_names)
 
                    with open("data/escalador.sav", "rb") as file:
                            x_scaler = pickle.load(file)
    
                    df_usu = x_scaler.transform(df_usuario)
    
                    # st.dataframe(df_usu)
    
                    with open("data/MLP.sav", "rb") as file:
                            model = pickle.load(file)
    
                    prediction = model.predict(df_usuario)
                    if prediction == 1:
                           st.snow()
                           st.info("Según nuestro modelo es probable que el cliente **cancele** su reserva.")
                    
                    else:
                           st.balloons()
                           st.info("Según nuestro modelo **se confirma** la reserva realizada")
    
if __name__ == "__main__":
                main()