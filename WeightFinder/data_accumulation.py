from typing import Dict, List

import pandas as pd


# Note: update macOS ssl certificates or you will get an error during the read_csv function
# /Applications/Python\ 3.6/Install\ Certificates.command

class DataAcc:

    def __init__(self):
        self.root_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data" \
                        "/csse_covid_19_daily_reports_us"
        self.data: Dict[str, List] = {}

    def pull_data(self, start_day, end_day, fields):
        date_range = pd.date_range(start=start_day, end=end_day)
        for date in date_range:
            f_date = date.strftime("%m-%d-%Y")
            try:
                url = self.root_url + "/" + f_date + ".csv"
                self.data[f_date] = pd.read_csv(url, error_bad_lines=False)[fields]
            except Exception as e:
                print(e)
                print("Error loading:", f_date)
        # print(len(self.data))

        return (len(self.data), list(self.data.keys()))

    def get_day(self, day):
        try:
            return self.data[day]
        except Exception:
            raise ValueError("Couldn't find the provided date in the dataset.")

    def get_state(self,state):
        state_acc = pd.DataFrame()
        for date,table in self.data.items():
            try:
                row = table.loc[table['Province_State'] == state]
                state_acc = state_acc.append(row)
            except Exception:
                print(state,"not found on",date)
                continue
        return state_acc

# Example calls to this class:
# data_acc = DataAcc()
# data_acc.pull_data("04-12-2020","11-30-2020")
# data_acc.get_day("04-13-2020")
