# CS4100_FinalProject

## Algorithm 1: Multiple Regression
***Calculate the 'C' coefficients of:***

f( C1(*variable1*), C2(*variable2*), ..., Cn(*variablen*) ) = *regressand_variable*

Examples:

f( C1(Hospitalization Rate) ) = Mortality_Rate

f( C1(Tests_Given), C2(Previous_Case_Number) ) = Case_Number

## Algorithm 2: Neural Network regression
***???***

## Algorithm 3: Search
***Search for `some anomaly` across all daily data:***
1. There are 216 days of COVID data recorded
2. Each day is stored in a .csv with 50 rows, each row being a U.S. state's COVID data for that day
3. 216 x 50 = 10800 *nodes* that need to be searched

***`some anomaly` could take many forms:***
1. a spike in some variable
2. a statistical anomaly (some unexpected combination of variable values)
3. a local/global maximum

