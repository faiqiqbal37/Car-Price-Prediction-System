import pandas as pd
import numpy as np
import streamlit as st
import pickle





model = pickle.load(open('model.sav','rb'))

st.title('Used Car Price Prediction App')
st.sidebar.header("Vehicle Information")

# FUNCTION
def user_report():
  CarName = st.sidebar.selectbox("Vehicle name:",('alfa-romeo', 'audi', 'bmw' ,'chevrolet' ,'dodge', 'honda' ,'isuzu' ,'jaguar',
 'mazda' ,'buick', 'mercury' ,'mitsubishi' ,'nissan', 'peugeot' ,'plymouth',
 'porsche', 'renault', 'saab', 'subaru', 'toyota' ,'volkswagen', 'volvo'))
  if (CarName == 'alfa-romeo'):
      CarName = 0
  if (CarName == 'audi'):
      CarName = 1
  if (CarName == 'bmw'):
      CarName = 2
  if (CarName == 'chevrolet'):
      CarName = 4
  if (CarName == 'dodge'):
      CarName = 5
  if (CarName == 'honda'):
      CarName = 6
  if (CarName == 'isuzu'):
      CarName = 7    
  if (CarName == 'jaguar'):
      CarName = 8
  if (CarName == 'mazda'):
      CarName = 9
  if (CarName == 'buick'):
      CarName = 3  
  if (CarName == 'mercury'):
      CarName = 10
  if (CarName == 'mitsubishi'):
      CarName = 11  
  if (CarName == 'nissan'):
      CarName = 12
  if (CarName == 'peugeot'):
      CarName = 13  
  if (CarName == 'plymouth'):
      CarName = 14
  if (CarName == 'porsche'):
      CarName = 15
  if (CarName == 'renault'):
      CarName = 16    
  if (CarName == 'saab'):
      CarName = 17    
  if (CarName == 'subaru'):
      CarName = 18    
  if (CarName == 'toyota'):
      CarName = 19
  if (CarName == 'volkswagen'):
      CarName = 20  
  if (CarName == 'volvo'):
      CarName = 21
    
  FuelType = st.sidebar.selectbox("Please Enter The Type Of Fuel", ('gas','diesel'), key = 'FuelType')
  if(FuelType == 'gas'):
      FuelType = 1
  else:
      FuelType = 0
      
  aspiration = st.sidebar.selectbox("Please Enter The Aspiration Type:", ('std','turbo'), key = 'aspiration' )
  if(aspiration == 'std'):
      aspiration = 0
  else:
      aspiration = 1
      
  doornumber = st.sidebar.selectbox("Please Enter The Number Of Doors:", ("Two", "Four"), key = 'doornumber') 
  if(doornumber == 'two'):
      doornumber = 1
  else:
      doornumber = 0
      
  carbody = st.sidebar.selectbox("Please Enter The Type Of Car Body.", ("convertible", "hatchback", "sedan", "wagon", "hardtop"), key = 'carbody')
  if(carbody == 'convertible'):
      carbody = 0
  if(carbody == 'hatchback'):
      carbody = 2
  if(carbody == 'sedan'):
      carbody = 3    
  if(carbody == 'wagon'):
      carbody = 4
  if(carbody == 'hardtop'):
      carbody = 1
      
  drivewheel = st.sidebar.selectbox("Please Mention The Drive Terrain Of The Car", ('rwd','fwd','4wd'), key = 'drivewheel'  )    
  if(drivewheel == 'rwd'):
      drivewheel = 2
  if(drivewheel == 'fwd'):
      drivewheel = 1
  if(drivewheel == '4wd'):
      drivewheel = 0
  
  cylindernumber = st.sidebar.selectbox("Please Enter The Number Of Cylinders.", ("two",'three', 'four', 'five','six', 'eight', 'twelve'), key = 'cylindernumber')
  if(cylindernumber == 'four'):
      cylindernumber = 2
  if(cylindernumber == 'six'):
      cylindernumber = 3
  if(cylindernumber == 'five'):
      cylindernumber = 1
  if(cylindernumber == 'three'):
      cylindernumber = 4
  if(cylindernumber == 'twelve'):
      cylindernumber = 5
  if(cylindernumber == 'two'):
      cylindernumber = 6
  if(cylindernumber == 'eight'):
      cylindernumber = 0
      
  
  carlength = st.sidebar.slider('Car Length', 50,400, 1)
  carwidth = st.sidebar.slider('Car Width', 50,100, 1)
  carheight = st.sidebar.slider('Car Height', 20,100, 1)
  enginesize = st.sidebar.slider('Engine Size', 50,500, 1)
  horsepower = st.sidebar.slider('Horsepower', 50,1000, 1)
  citympg = st.sidebar.slider('City MPG', 1,100, 1)
  highwaympg = st.sidebar.slider('Highway MPG', 1,100, 1)
  
                                      
    
    
  
  user_report_data = {
      'CarName':CarName,
      'FuelType':FuelType,
      'aspiration':aspiration,
      'doornumber':doornumber,
      'carbody':carbody,
      'drivewheel':drivewheel,
      'cylindernumber':cylindernumber,
      'carlength':carlength,
      'carwidth':carwidth,
      'carheight':carheight,
      'enginesize':enginesize,
      'horsepower':horsepower,
      'citympg':citympg,
      'highwaympg':highwaympg
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data

user_data = user_report()


st.header('Vehicle Data')
st.write(user_data)

if st.button("Predict"):
    salary = model.predict(user_data)
    st.subheader('Predicted Price')
    st.subheader('$'+str(np.round(salary[0], 2)))

