from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import numpy as np
import haversine as hs
from sklearn.decomposition import PCA
from datetime import datetime
import pandas as pd


class DataProcessor:
    def __init__(self, train):
        self.train = train

    def convert_time(self, row):
        _, lat, lon, year, week = row.split('_')
        year, week = int(year), int(week)
        
        base_date = datetime(year, 1, 1) + timedelta(days=(week) * 7)
        return base_date.strftime('%Y-%m-%d')

    def get_month(self, df):
        df['month'] = pd.to_datetime(df['date']).dt.month
        return df

    def covid_flag(self, df):
        
        df['date'] = pd.to_datetime(df['date'])
        

        df['year'] = df['date'].datetime.year
        df['week_no'] = df['date'].datetime.isocalendar().week
        
        
        df['covid_flag'] = np.where((df['year'] == 2020) & 
                                    ((df['year'] != datetime.now().year) | (df['week_no'] >= 12)) &
                                    (df['week_no'] <= 26), 1, 0)
        
        return df

    def cyclical_encoding(self, df):

        df['week_no_sin'] = np.sin(df['week_no'] * (2 * np.pi / 52))
        df['week_no_cos'] = np.cos(df['week_no'] * (2 * np.pi / 52))
        return df

    def rotation(self, df):
        df['rot_45_x'] = (0.707 * df['latitude']) + (0.707 * df['longitude'])
        df['rot_45_y'] = (0.707 * df['longitude']) + (0.707 * df['latitude'])
        df['rot_30_x'] = (0.866 * df['latitude']) + (0.5 * df['longitude'])
        df['rot_30_y'] = (0.866 * df['longitude']) + (0.5 * df['latitude'])
        return df

    def pca(self, df):
        coordinates = df[['latitude', 'longitude']].values
        pca_obj = PCA().fit(coordinates)
        df['pca_x'] = pca_obj.transform(df[['latitude', 'longitude']])[:, 0]
        df['pca_y'] = pca_obj.transform(df[['latitude', 'longitude']])[:, 1]
        return df

    def preprocess(self, df):

        # # Handle missing values in both train and test datasets
        # missing_ratios_train = self.train.isnull().mean()
        # columns_to_drop_train = missing_ratios_train[missing_ratios_train > 0.9].index
        # self.train = self.train.drop(columns_to_drop_train, axis=1)
        # self.train = self.train.fillna(self.train.mean())

        
        df = self.cyclical_encoding(df)

        
        # cluster_df = self.train.groupby(by=['latitude', 'longitude'], as_index=False)['emission'].mean()
        # model = KMeans(n_clusters=5)
        # cluster_df['kmeans_group'] = model.fit_predict(cluster_df)

        df = self.get_month(df)

        df = self.rotation(df)
        # df = self.pca(df)

       
        # # Create distance to the highest emission location column
        # max_lat_lon_emission = cluster_df.loc[cluster_df['emission'] == cluster_df['emission'].max(),
        #                                       ['latitude', 'longitude']]
        
        latitude = 2.378
        longitude = 29.222  
       
        df['distance_to_max_emission'] = hs.haversine(
            (df['latitude'].values, df['longitude'].values),
            (latitude, longitude)
        )

        # return df.to_dict(orient='records')
        # # Emission_zero column
        # cluster_df['emission_zero'] = cluster_df['emission'].apply(lambda x: 1 if x == 0 else 0)

        # # Merge df and cluster_df
        # df = df.merge(cluster_df[['latitude', 'longitude', 'kmeans_group', 'emission_zero']],
        #               on=['latitude', 'longitude'])

        return df

