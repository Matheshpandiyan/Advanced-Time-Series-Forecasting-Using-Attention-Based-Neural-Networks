
import numpy as np
import pandas as pd

def generate_data(n=500):
    t = np.arange(n)
    x1 = np.sin(0.02*t) + 0.1*np.random.randn(n)
    x2 = np.cos(0.015*t) + 0.1*np.random.randn(n)
    y = 0.5*x1 + 0.3*x2 + 0.2*np.sin(0.05*t) + 0.1*np.random.randn(n)
    df = pd.DataFrame({'x1':x1,'x2':x2,'y':y})
    df.to_csv('series.csv', index=False)

if __name__=='__main__':
    generate_data()
