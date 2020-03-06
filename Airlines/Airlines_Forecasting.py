import pandas as pd
Airlines = pd.read_excel("C:/Training/Analytics/Forecasting/Airlines/Airlines+Data.xlsx")
month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 
import numpy as np

# Creating a Date column to store the actual Date format for the given Month column
Airlines["Date"] = pd.to_datetime(Airlines.Month,format="%b-%y")

Airlines["month"] = Airlines.Date.dt.strftime("%b") # month extraction
#Airlines["Day"] = Airlines.Date.dt.strftime("%d") # Day extraction
#Airlines["wkday"] = Airlines.Date.dt.strftime("%A") # weekday extraction
Airlines["year"] = Airlines.Date.dt.strftime("%Y") # year extraction

# =============================================================================
# p = Airlines["Month"][0]
# p[0:3]
# Airlines['months']= 0
# 
# for i in range(159):
#     p = Airlines["Month"][i]
#     Airlines['months'][i]= p[0:3]
# =============================================================================

Airlines1 = Airlines.drop(['Month','Date','year'],axis=1)

month_dummies = pd.DataFrame(pd.get_dummies(Airlines['month']))

Airlines1 = pd.concat([Airlines1,month_dummies],axis = 1)

Airlines1["t"] = np.arange(1,97)

Airlines1["t_squared"] = Airlines1["t"]*Airlines1["t"]
Airlines1.columns
Airlines1["Log_Passengers"] = np.log(Airlines1["Passengers"])

Airlines1.Passengers.plot()
Train = Airlines1.head(80)
Test = Airlines1.tail(16)

# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,13))

####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear

##################### Exponential ##############################

Exp = smf.ols('Log_Passengers~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#################### Quadratic ###############################

Quad = smf.ols('Passengers~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad

################### Additive seasonality ########################

add_sea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea

################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Passengers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('Log_Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

##################Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('Log_Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse

# so rmse_add_sea has the least value among the models prepared so far 
# Predicting new values 
#
##predict_data = pd.read_csv("E:\Bokey\Excelr Data\Python Codes\Forecasting_Python\Predict_new.csv")
#model_full = smf.ols('Log_Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Test).fit()
#
#pred_new  = pd.Series(add_sea_Quad.predict(predict_data))
#pred_new
#
#predict_data["forecasted_Passengers"] = pd.Series(pred_new)
