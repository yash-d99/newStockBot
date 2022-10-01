import pandas as pd
import yfinance as yf
import os, contextlib


def getData(symbols):
  offset = 0
  limit = 3000
  period = 'max' # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max

  limit = limit if limit else len(symbols)
  end = min(offset + limit, len(symbols))
  is_valid = [False] * len(symbols)
  # force silencing of verbose API
  with open(os.devnull, 'w') as devnull:
      with contextlib.redirect_stdout(devnull):
          for i in range(offset, end):
              s = symbols[i]
              data = yf.download(s, period=period)
              if len(data.index) == 0:
                  continue
          
              is_valid[i] = True
              data.to_csv('hist/{}.csv'.format(s))

  #print('Total number of valid symbols downloaded = {}'.format(sum(is_valid)))