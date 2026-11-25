"""
Notes for Fahmi

This file will be used to take data from whatever source we have and convert it to a pandas df.
If you *reeeeeally* want to be fancy, you can also denormalise the tables that we get so that 
models.py can just become a plug and play. As for how you want to denormalise, up to you. I can adjust 
models accordingly if I have to.


Leave the data_loader_csv file as it is, that isn't gonna be that important anyway unless you reckon
we can include that in the report to show what that simple form of storage would look like (which 
is a shout tbh). Otherwise I think it's just postgres and denormalisation? 

Just don't change the data, only change (if you think it is a good idea) how it is stored in the df.
"""

import pandas as pd

def data_loader_csv(path = r"ML\smartgrid_5yr_hourly_natural-selected.csv"):
    df = pd.read_csv(path)

    return df

if __name__ == "__main__":
    print(data_loader_csv())