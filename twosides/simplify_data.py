import os
import pandas as pd
twosides_500_old = pd.read_csv('data/twosides_ge_500.csv',delimiter=',')
twosides_500_new = twosides_500_old[:50000]
twosides_500_new.to_csv('data/twosides_ge_500_new.csv',index=False)
print(twosides_500_new)
