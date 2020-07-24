import pandas as pd 
import random
from datetime import datetime, timedelta
from datetime import date


#generate random dates
def gen_datetime(min_year=2016, max_year=2018):
    # generate a datetime yyyy-mm-dd hh:mm:ss.000000
    start = datetime(min_year, 1, 1, 00, 00, 00)
    years = max_year - min_year + 1
    end = start + timedelta(days=365 * years)
    return start + (end - start) * random.random()


#get the numerical day of week
def get_dayOfWeek(yearA ,monthA , dayA):
    
    #generate a date objet that has the numerical weekday
    dateAux = date(year=yearA, month=monthA, day=dayA)
    #indice do dia da semana de 0 a 6 = segunda a domingo\n",
    numericalDay = dateAux.weekday()
    
    return numericalDay + 1

#generate random coordinates, given the max and min coordinates present on the bostom crime dataset
def gen_coordinates():
    min_longitude = -71.17867378
    max_longitude = -70.96436489
    min_latitude = 42.2324133
    max_latitude = 42.39504158
    rand_latitude = random.uniform(min_latitude,max_latitude)
    rand_longitude = random.uniform(min_longitude,max_longitude)
    return (rand_latitude,rand_longitude)

def get_district():
    districts = ['B3', 'E18', 'E5', 'A1', 'D4', 'C11', 'B2', 'E13', 'C6', 'D14', 'A7', 'A15']
    index = random.randint(0,len(districts)-1)
    return districts[index]


    
#generate no-occurrence entries for the dataset    
def gen_syntheticData(tamAmostra):
    #dictionary with the coluns of the dataset
    data = {"OFFENSE_CODE_GROUP" : [],"DISTRICT" : [],"YEAR" : [],"MONTH" : [],"DAY_OF_WEEK" : [],"HOUR" : [],"Lat" : [],"Long" : []}
    
    for i in range(tamAmostra):
        dateAux = gen_datetime()
        year = dateAux.year
        month = dateAux.month
        day = dateAux.day
        coordinate = gen_coordinates()
        
        data["OFFENSE_CODE_GROUP"].append("not occurred")
        data['DISTRICT'].append(get_district())
        data['YEAR'].append(year)
        data['MONTH'].append(month)
        data['DAY_OF_WEEK'].append(get_dayOfWeek(year,month,day))
        data['HOUR'].append(dateAux.hour)
        data['Lat'].append(coordinate[0])
        data['Long'].append(coordinate[1])
        
        
    df = pd.DataFrame(data)
    return df