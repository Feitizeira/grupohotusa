# Grupo Hotusa - Data Scientist Test

El director de Revenue Management nos comenta que las cancelaciones tienen un impacto negativo en los resultados del hotel y se pregunta si con todo esto del Machine Learning y la Inteligencia Artificial seríamos capaces de predecir qué reservas van a cancelar y cuáles no. 

El director quiere poner en marcha este modelo para poder realizar overbooking (llenar el hotel por encima del inventario total disponible con el objetivo de ocupar aquellas reservas que se van a cancelar). El hotel está en el centro de Lisboa y en la misma ciudad el grupo hotelero tiene varios hoteles, por lo que **no hay ningún riesgo en llenar el hotel más de la cuenta** y enviar el exceso de reservas a otros hoteles en la ciudad.

## Descripción de la tarea

El dataset (**hotusa_cancellations.csv**) con el que trabajar contiene las siguientes columnas:
- *HotelId*: Id del hotel
- *ReservationStatusDate*: Fecha en la que se realizó la reserva
- *ArrivalDate*: Fecha de inicio de la reserva
- *LeadTime*: Antelación (días entre la generación de la reserva hasta la fecha de inicio de ésta)
- *StaysInWeekendNights*: Número de noches entre semana
- *StaysInWeekNights*: Número de noches el fin de semana
- *Adults*: Número de adultos en la reserva
- *Children*: Número de niños en la reserva
- *CustomerType*: Tipo de cliente
- *ADR*: Average Daily Rate (precio medio por reserva. Precio total de la reserva dividido por el número de noches)
- *Meal*: Tipo de Alojamiento (BB (Bed and Breakfast), HB (Half Board), Undefined, FB (Full Board), and SC (Self Catering))
- *Country*: País de procedencia del cliente
- *Company*: Compañía del cliente
- *ReservedRoomType*: Habitación reservada
- *IsRepeatedGuest*: Cliente repetitivo
- *IsCanceled*: Booleano que indica si la reserva se canceló o no (TARGET variable)

Debes crear un report reproducible (en un notebook en Python o R) siguiendo estos pasos:
1. **EDA sencillo y básico** (A Revenue Management le preocupan más los resultados que los insights)
  * ¿Qué variables parecen ser las más predictivas?

2. **Preparación de los datos**
  * ¿Le darías el mismo tratamiento a todas las variables?
  * ¿Utilizarías todas las variables o eliminarías alguna de ellas por no ofrecer valor?
  
3. **Construcción del modelo**
  * ¿Qué tipo de modelo es el más adiente? Support Vector Machines, Decision Tree, Logistic Regression, K-Means, Redes Neuronales, Random Forest, Gradient Boosting, Naive Bayes, PCA, Lasso Regression...
  * No es necesario probar entre cientos de modelos con distintos hiperparámetros para saber cuál da mejores resultados, basta con justificar la elección.
  
4. **Evaluación del modelo**
  * ¿Cuál sería el benchmark a batir? ¿Estamos contentos con los resultados obtenidos?
  * ¿Qué resultados podríamos esperar con este modelo en producción?

## Entrega

Entrega tus resultados en notebook o en un link al repo en GitHub a: carlos.perez@eurostarshotelcompany.com 
