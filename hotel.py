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
                                                    value = 2
                                )
                        Meal = st.selectbox(label = "Tipo de alojamiento", 
                                        options = ["SC", "BB", "HB", "FB", "Undefined"]
                                )   
                        Client = st.selectbox(label= "Tipo de cliente",
                                        options = ["Group", "Contract", "Transient", "Transient-Party"]
                                )
                        Mes_Estancia = st.slider(label= "Mes de la Estancia",
                                                min_value = 1,
                                                max_value= 12,
                                                step=1
                                )
                        Country = st.selectbox(label= "Código del país",
                                               options=['PRT', 'ISR', 'BRA', 'AUT', 'GBR', 'IRL', 'RUS', 'ESP', 'SVK',
                                                'SWE', 'GEO', 'FRA', 'CHE', 'DEU', 'ITA', 'USA', 'NO_COUNTRY',
                                                'CN', 'POL', 'NLD', 'MAR', 'LUX', 'BEL', 'HUN', 'ZAF', 'ROU',
                                                'PHL', 'AGO', 'NOR', 'NGA', 'AUS', 'ARG', 'HRV', 'MDV', 'PAK',
                                                'FIN', 'IDN', 'BLR', 'CZE', 'CHN', 'ARE', 'EST', 'LBN', 'GIB',
                                                'DNK', 'SVN', 'TUR', 'LVA', 'AZE', 'ALB', 'KOR', 'KWT', 'GRC',
                                                'URY', 'CHL', 'IND', 'OMN', 'MOZ', 'BWA', 'MEX', 'UKR', 'SMR',
                                                'PRI', 'SRB', 'LTU', 'CYM', 'ZMB', 'CPV', 'ZWE', 'DZA', 'CRI',
                                                'JAM', 'IRN', 'CAF', 'CYP', 'NZL', 'KAZ', 'THA', 'COL', 'DOM',
                                                'MKD', 'MYS', 'VEN', 'ARM', 'JPN', 'LKA', 'CUB', 'CMR', 'BIH',
                                                'MUS', 'COM', 'SUR', 'UGA', 'BGR', 'CIV', 'JOR', 'SYR', 'SGP',
                                                'BDI', 'SAU', 'AND', 'VNM', 'PLW', 'QAT', 'EGY', 'PER', 'MLT',
                                                'MWI', 'ECU', 'MDG', 'ISL', 'UZB', 'NPL', 'BHS', 'MAC', 'TGO',
                                                'TWN', 'HKG', 'DJI']
                                )
                                            
                with col2:
                        StaysInWeekendNights = st.number_input(label= "Noches en fin de semana de la reserva",
                                                value = 2
                                )
                        Children = st.number_input(label= "Número de niños",
                                                value = 0
                                )
                        Company = st.selectbox(label = "Código de la compañia", 
                                                options = ['0','10','103','104','108','109','110','112','113','116','118','12','120',
                                                        '126','130','135','137','14','144','146','148','154','159','16','165','167',
                                                        '168','169','174','178','184','192','193','195','20','200','203','204','207'
                                                        ,'212','22','223','224','225','232','240','242','246','250','251','254','255',
                                                        '263','268','269','270','272','274','277','278','28','281','282','286','287',
                                                        '289','29','290','291','292','297','302','307','308','31','312','317','318',
                                                        '319','32','323','324','325','329','330','331','333','337','338','34','342',
                                                        '343','346','347','349','351','353','355','356','358','360','361','362','364',
                                                        '365','366','367','369','370','371','372','376','377','378','379','380','382',
                                                        '384','388','39','390','391','392','394','395','396','397','398','399','40',
                                                        '400','401','402','403','405','407','408','409','410','413','415','416','419',
                                                        '42','421','422','423','424','425','428','43','435','436','437','439','442',
                                                        '443','444','445','447','448','454','455','456','457','458','459','460','461'
                                                        ,'465','466','47','470','477','482','484','485','487','490','491','494','496'
                                                        ,'498','499','501','504','506','507','51','511','512','513','514','515','516',
                                                        '518','52','520','521','523','525','528','53','530','534','539','54','541',
                                                        '543','59','6','61','62','64','72','78','80','81','82','83','84','86','88',
                                                        '9','92','94','99']
                                )
                        Mes_Reserva = st.slider(label= "Mes de la Reserva",
                                                min_value = 1,
                                                max_value= 12,
                                                step=1
                                )
                        IsRepeatedGuest = st.radio(label= "Es un cliente que repite",
                                        options = ("Si", "No"), 
                                        index = 1,
                                        disabled = False,
                                        horizontal = True,
                                )
                        
                with col3:
                        StaysInWeekNights = st.number_input(label= "Noches durante la semana",
                                                value = 0
                                )
                        Adults = st.number_input(label= "Número de adultos",
                                                value = 2
                                )  
                        ADR = st.number_input(label= "Precio medio de la reserva por día de estancia",
                                                value = 92
                                )
                        ReservedRoomType = st.slider(label= "Tipo de Habitación reservada",
                                                min_value = 0,
                                                max_value= 8,
                                                step=1
                                )
                         
                if st.form_submit_button("Enviar"):
                    datos = [LeadTime, StaysInWeekendNights, StaysInWeekNights, Adults, Children, ADR, 
                            Country, Company, ReservedRoomType, IsRepeatedGuest, Mes_Estancia, Mes_Reserva, Client, Meal]

                    for i in range(len(datos)):
                        if datos[i] == 'Si':
                            datos[i] = 1                  
                        elif datos[i] == 'No':
                            datos[i] = 0
                    dfdata = pd.DataFrame([datos], columns=['LeadTime','StaysInWeekendNights','StaysInWeekNights','Adults','Children',
                                                            'ADR','Country','Company','ReservedRoomType','IsRepeatedGuest','Mes_Estancia',
                                                            'Mes_Reserva','Contract','Group'])
                    if datos[12]=="Transient":
                        dfdata[['Contract','Group','Transient','Transient-Party']] = ["0", "0", "1", "0"]
                    elif datos[12]=="Transient-Party":
                        dfdata[['Contract','Group','Transient','Transient-Party']] = ["0", "0", "0", "1"]
                    elif datos[12]=="Group":
                        dfdata[['Contract','Group','Transient','Transient-Party']] = ["0", "1", "0", "0"]
                    elif datos[12]=="Contract":
                        dfdata[['Contract','Group','Transient','Transient-Party']] = ["1", "0", "0", "0"]
                    if datos[13]=="BB":
                        dfdata[['BB', 'FB', 'HB', 'SC', 'Undefined']] = ["1", "0", "0", "0", "0"]
                    elif datos[13]=="FB":
                        dfdata[['BB', 'FB', 'HB', 'SC', 'Undefined']] = ["0", "1", "0", "0", "0"]
                    elif datos[13]=="HB":
                        dfdata[['BB', 'FB', 'HB', 'SC', 'Undefined']] = ["0", "0", "1", "0", "0"]
                    elif datos[13]=="SC":
                        dfdata[['BB', 'FB', 'HB', 'SC', 'Undefined']] = ["0", "0", "0", "1", "0"]
                    elif datos[13]=="Undefined":
                        dfdata[['BB', 'FB', 'HB', 'SC', 'Undefined']] = ["0", "0", "0", "0", "1"]

                    dfo["Country"] = dfo["Country"].fillna("NO_COUNTRY")

                    # Importar el encoder:
                    with open("data/country.sav", "rb") as file:
                        encoder = pickle.load(file)
                    numeracion = encoder.transform(dfo[["Country"]])
                    numeros = pd.DataFrame(numeracion, columns=["Country_Code"])
                    nombres = dfo["Country"]
                    paises = pd.concat([nombres,numeros], axis=1)
                    codigo = paises[paises["Country"]==datos[6]]
                    dfdata["Country"] = codigo.iloc[0, 1]
 
                    with open("data/escalador.sav", "rb") as file:
                            x_scaler = pickle.load(file)
    
                    df_scaled = x_scaler.transform(dfdata)
    
                    with open("data/MLP.sav", "rb") as file:
                        model = pickle.load(file)
    
                    prediction = model.predict(df_scaled)
                    if prediction == 1:
                           st.snow()
                           st.info("Según nuestro modelo es probable que el cliente **cancele** su reserva.")
                    
                    else:
                           st.balloons()
                           st.info("Según nuestro modelo **se confirma** la reserva realizada")

                    col4, col5 = st.columns(2)

                    with col4:
                        st.write("% probabilidad de confirmar la reserva:")
                        st.write(int(model.predict_proba(df_scaled)[0][0]*100))

                    with col5:
                        st.write("% probabilidad de cancelar la reserva:")
                        st.write(int(model.predict_proba(df_scaled)[0][1]*100))
if __name__ == "__main__":
                main()