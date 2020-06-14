import akshare as ak
epidemic_hist_province_df = ak.covid_19_
print(epidemic_hist_province_df)

outputpath='COVID19.csv'
epidemic_hist_province_df.to_csv(outputpath,sep=',',index=True,header=True)