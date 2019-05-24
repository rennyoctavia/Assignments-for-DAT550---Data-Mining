import pandas as pd
import numpy as np
from sklearn import preprocessing

def clean_data(filepath_train, filepath_test):

    house_price_test = pd.read_csv(filepath_test,encoding = 'latin-1')
    house_price_train = pd.read_csv(filepath_train,encoding = 'latin-1')

    house_price_train_column = house_price_train.copy()

    target_train = house_price_train['SalePrice']

    #house_price_test.drop_duplicates(inplace = True)
    house_price_train.drop_duplicates(inplace = True)

    
    test_id = house_price_test['Id']
    #There is no duplicates of Id and order which shows it is only a sequence of number, so we drop PID and Order
    house_price_test = house_price_test.drop(columns='Id')
    house_price_train = house_price_train.drop(columns = 'Id')
    house_price_train_column.drop(columns='Id',inplace=True)
    


    ########################################################################

    no_facility = ['Alley','FireplaceQu','GarageFinish','GarageType','GarageQual','GarageCond','PoolQC','Fence','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']

    for name in no_facility:
        house_price_train[name]=house_price_train[name].fillna('NA')
        house_price_test[name] = house_price_test[name].fillna('NA')
        # To score them, we groupby by the column name and categorize them by the mean of sale price for each unique value
        # sort the unique value by the mean of sale price from low to high and keep them in the list
        grouped = house_price_train.groupby(name)['SalePrice'].mean().reset_index().sort_values(by='SalePrice').reset_index(drop=True)
        replace_list = list(grouped[name])
        # for each value in the columns, replace with the index from the list(which shows the rank/score), for each value.
        house_price_train[name]=house_price_train[name].apply(lambda x: replace_list.index(x))
        house_price_test[name] = house_price_test[name].apply(lambda y: replace_list.index(y))

    #########################################################################

    #convert categorical to numerical manually for columns with ranking 
    cleanup_dicts= {'ExterQual':    {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa':1, 'Po':0},
                    'ExterCond':    {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa':1, 'Po':0},
                    'Functional' :  {'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod':4, 'Maj1':3, 'Maj2':2,'Sev': 1, 'Sal': 0},
                    'SaleCondition':{'Normal': 5, 'Abnorml':4, 'AdjLand':3,'Alloca':2,'Family':1,'Partial': 0},
                    'HeatingQC':    {'Normal': 5, 'Abnorml':4, 'AdjLand':3,'Alloca':2,'Family':1,'Partial': 0},
                    'LotShape' :    {'Reg':3,'IR3':2,'IR2':1,'IR1':0},
                    'Utilities':    {'AllPub':3,'NoSewr':2,'NoSeWa':1,'ELO':0},
                    'HeatingQC':   {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa':1, 'Po':0},
                    'KitchenQual': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa':1, 'Po':0}           
    }

    house_price_train.replace(cleanup_dicts, inplace = True)
    house_price_test.replace(cleanup_dicts, inplace= True)

    #########################################################################

    concat_house_price = pd.concat([house_price_train.drop(columns='SalePrice'),house_price_test])
    
    #########################################################################    
    
    #Filling in columns with which values have no ranking, to be converted to integer 0,1,2,...

    column_norank = ['MSZoning','Street','LandSlope','Neighborhood','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st',
                     'Exterior2nd','Foundation','Heating','CentralAir','Electrical','PavedDrive','SaleType','LandContour','LotConfig',
                     'MasVnrType','Condition1','Condition2','MiscFeature']

    #make subset of dataframe with all columns which have values that cannot be ranked
    df_norank= concat_house_price.loc[:,column_norank]

    #Change the values for each column to integer 0,1,2,3 .... n, with n is the number of unique value per column
    for column in column_norank:
        df_norank[column]=df_norank[column].fillna('NA')
        count_unique = len(df_norank[column].unique())
        temp_unique = df_norank[column].unique()
        for i in range(count_unique):
            df_norank[column][df_norank[column]==temp_unique[i]] = i

    #Apply above in the concatenated dataframe        
    concat_house_price.loc[:,column_norank] = df_norank.loc[:,column_norank]

    #########################################################################    

    ##Fill in GarageCars and Garage Area with 0, as the status of GarageType is NA
    #The reasone is similar as above, the Nan values means no facility available, therefore as this is numerical, fill in Nans for both columns with 0.
    concat_house_price['GarageCars']=concat_house_price['GarageCars'].fillna(0)
    concat_house_price['GarageArea']=concat_house_price['GarageArea'].fillna(0)

    ##The rest of Nan Values for Lot Frontage, GarageYrBuilt will be fill in  by the median value


    #########################################################################  

    #MasVnrType and MasVnrArea
    #fill in with None for the NA, as one of the category of Mas VnrType is None, and for category NOne, the MasVnrArea is 0
    concat_house_price['MasVnrType']=concat_house_price['MasVnrType'].fillna('None')
    concat_house_price['MasVnrArea']=concat_house_price['MasVnrArea'].fillna(0)

    #There is inconsistency with the MasVnrType, None, some of it have value > 0
    temp_df=concat_house_price[concat_house_price.MasVnrType=='None']
    MasVnrNone = temp_df.loc[:,['MasVnrType','MasVnrArea']].pivot_table(index=['MasVnrType','MasVnrArea'],aggfunc='count')

    #print('MasVnrNone have MasVnrArea values as follows : ',MasVnrNone)

    #as wee see that None should have value 0, for the one have value 1 is possbily typo, so we replace with 0
    concat_house_price.loc[(concat_house_price.loc[:,'MasVnrArea'] == 1) & (concat_house_price.loc[:,'MasVnrType'] == 'None'),'MasVnrArea'] = 0

    #For MasVnrType None, with MasVnrArea > 0, we change the MasVnrType to the mode() of the MasVnrType (as it should not be NONE)
    concat_house_price.loc[(concat_house_price.loc[:,'MasVnrArea'] != 0) & (concat_house_price.loc[:,'MasVnrType'] == 'None'),'MasVnrType'] =concat_house_price.loc[concat_house_price.loc[:,'MasVnrType'] != 'None','MasVnrType'].mode()[0]

    #########################################################################  
    
    # Fill in the NAN with the median values
    concat_house_price.fillna(concat_house_price.median(),inplace=True)  

    # Checking the summary of dataframe for null values
    concat_house_price.isnull().sum().sum()

    #########################################################################

    #After converting all columns, split the data into train and test again

    n_train = house_price_train.shape[0]
    n_test = house_price_test.shape[0]
    house_price_train.iloc[:,house_price_train.columns != 'SalePrice'] = concat_house_price.iloc[0:n_train,:]
    house_price_test = concat_house_price.iloc[n_train:n_train+n_test,:]

    #########################################################################

    #create new feature total size of the houses which probably correlates better
    house_price_train['TotalSize'] = house_price_train['TotalBsmtSF']+ house_price_train['GrLivArea']
    house_price_train_column['TotalSize'] = house_price_train_column['TotalBsmtSF']+ house_price_train_column['GrLivArea']
    house_price_test['TotalSize'] = house_price_test['TotalBsmtSF']+ house_price_test['GrLivArea']
    
    #########################################################################

    ##Create new Feature called Total Bathroom Bsmt Full Bath	Bsmt Half Bath	Full Bath	Half Bath
    house_price_train['TotalBathroom'] =house_price_train['BsmtFullBath']+house_price_train['BsmtHalfBath']/2+house_price_train['FullBath']+house_price_train['HalfBath']/2 
    house_price_train_column['TotalBathroom'] =house_price_train_column['BsmtFullBath']+house_price_train_column['BsmtHalfBath']/2+house_price_train_column['FullBath']+house_price_train_column['HalfBath']/2 
    house_price_test['TotalBathroom'] =house_price_test['BsmtFullBath']+house_price_test['BsmtHalfBath']/2+house_price_test['FullBath']+house_price_test['HalfBath']/2 

    #########################################################################

    #Move the SalePrice to the last Column
    house_price_train.drop('SalePrice',axis=1,inplace=True)
    house_price_train['SalePrice'] = target_train

    house_price_train_column.drop('SalePrice',axis=1,inplace=True)
    house_price_train_column['SalePrice'] = target_train

    #########################################################################
    #set number of columns to drop (based on smallest correlation)
    num_drop_column = 5
    #Drop number of columns starting from the smallest correlation
    columns_to_drop = list(abs(house_price_train.corrwith(house_price_train['SalePrice'],axis=0)).sort_values(axis=0, ascending=True).index[0:num_drop_column])
    house_price_train.drop(columns_to_drop,axis=1,inplace=True)
    house_price_train_column.drop(columns_to_drop,axis=1,inplace=True)

    house_price_test.drop(columns_to_drop,axis=1, inplace=True)

    #########################################################################
    #Get the type of each Attribute
    columns_types=house_price_train_column.dtypes.astype(str)
    dict_types = {'int64':'num','float64':'num','object':'cat'}
    columns_types = columns_types.apply(lambda x : dict_types[x])
    columns_types['MSSubClass'] = 'cat'
    columns_types['OverallQual'] = 'cat'
    columns_types['OverallCond'] = 'cat'
    
    return house_price_train, house_price_test, columns_types,test_id

