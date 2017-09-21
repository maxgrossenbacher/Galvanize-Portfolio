import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

if __name__ == '__main__':
    mpl.rcParams.update({
        'font.size': 16.0,
        'axes.titlesize': 'large',
        'axes.labelsize': 'medium',
        'xtick.labelsize': 'small',
        'ytick.labelsize': 'small',
        'legend.fontsize': 'small',
    })

    # Force pandas to display all columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # Read in data
    df = pd.read_json('data/data.json')

    # Check the overall shape of the data
    df.shape

    # Take a subsample if large data is running slowly
    df = df.sample(frac=.15, random_state=42)
    df = df.reset_index()

    # Creating feature for number of previous events
    df['num_previous_events'] = [len(x) for x in df['previous_payouts']]

    # Creating feature for total previous payouts
    final = []
    for x in df['previous_payouts']:
        temp = []
        for d in x:
            temp.append(d['amount'])
        final.append(sum(temp))
    df['total_previous_payouts'] = final

    # Checking percentage of null values
    nulls = df.isnull().sum() / float(df.shape[0])
    nulls.sort_values(ascending=False)

    # Drop any columns with 25% or more null values
    for col in df:
        if df[col].isnull().sum() / float(df.shape[0]) >= 0.25:
            df.drop(col, axis=1, inplace=True)

    # Creating boolean target variable
    target_list = []
    for x in df['acct_type']:
        if x == 'premium':
            target_list.append(0)
        else:
            target_list.append(1)
    df['target'] = target_list

    # Check if target classes are evenly sized
    # A split greater than 70/30 requires techniques for rebalancing
    df.target.value_counts() / len(df)

    # Drop some useless columns
    df.drop(['index', 'object_id', 'venue_longitude', 'venue_longitude', 'venue_latitude', 'acct_type',
             'previous_payouts', 'venue_address', 'ticket_types'], inplace=True, axis=1)

    # Set numerical & categorical values for use later
    numerical_vals = df.select_dtypes(exclude=['object', 'bool'])
    categorical_vals = df.select_dtypes(include=['object', 'bool'])

    # Checking unique values in categorical columns
    lengths = []
    for col in categorical_vals.columns:
        lengths.append([col, len(categorical_vals[col].unique())])
    lengths

    # Drop columns with zero variance or change to objects if coded incorrectly
    for col in numerical_vals:
        print(col)
        print(numerical_vals[col].var())

    # Change any data types that are incorrect
    df.show_map = df.show_map.astype('object')
    df.has_logo = df.has_logo.astype('object')
    df.has_analytics = df.has_analytics.astype('object')
    df.fb_published = df.fb_published.astype('object')
    df.delivery_method = df.delivery_method.astype('object')

    # Creating boolean variable for country
    df['US'] = df['country'] == 'US'

    # Changing date strings to readable format
    df.event_created = pd.to_datetime(df.event_created, unit='s')
    df.event_end = pd.to_datetime(df.event_end, unit='s')
    df.event_published = pd.to_datetime(df.event_published, unit='s')
    df.event_start = pd.to_datetime(df.event_start, unit='s')
    df.user_created = pd.to_datetime(df.user_created, unit='s')

    # Creating new variables for time related measurements
    df['hours_between_published_and_created'] = [i.seconds /
                                                 3600.0 for i in (df.event_published - df.event_created)]
    df['hours_between_event_start_and_end'] = [
        i.seconds / 3600.0 for i in (df.event_end - df.event_start)]
    df['hour_of_day_event_published'] = [i.hour for i in df.event_published]

    # Dropping more columns
    df.drop(['approx_payout_date', 'event_created', 'email_domain',
             'event_end', 'event_start', 'event_published', 'user_created'], inplace=True, axis=1)

    # Creating new feature based on all caps name
    final = []
    for x in df['name']:
        if x == '':
            final.append(False)
        else:
            final.append(x[len(x) - 1].istitle())
    df['last_letter_name_caps'] = final

    # Creating new feature based on all lowercase name
    final = []
    for x in df['name']:
        if x == '':
            final.append(False)
        else:
            final.append(x[0].islower())
    df['first_letter_name_lowercase'] = final

    # Creating features for blank info
    df['desc_blank'] = df.org_desc == ''
    df['org_name_blank'] = df.org_name == ''
    df['payee_blank'] = df.payee_name == ''
    df['venue_name_blank'] = df.venue_name == 'Missing'
    df['venue_state_blank'] = df.venue_state == 'Missing'
    df['payout_type_blank'] = df.payout_type == ''

    # Reset values
    numerical_vals = df.select_dtypes(exclude=['object', 'bool'])
    categorical_vals = df.select_dtypes(include=['object', 'bool'])

    # Temp fill numerical null values
    for col in numerical_vals:
        df[col].fillna(df[col].mean(), inplace=True)

    # Temp fill categorical null values
    df.fillna('Missing', inplace=True)

    # Creating new feature to combine sale duration columns
    df['max_sale_duration'] = df[['sale_duration', 'sale_duration2']].max(axis=1)

    # Boxplots of individual columns
    for col in numerical_vals:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        ax.set_title(col)
        sns.violinplot(x=df[col], orient='v', ax=ax, palette='pastel')
        text = '75th Percentile: {}\nMedian: {}\n25th Percentile: {}'.format(np.percentile(df[col], 75),
                                                                             np.median(df[col]), np.percentile(df[col], 25))
        at = AnchoredText(text, prop=dict(size=15), frameon=True, loc=1)
        ax.add_artist(at)
        plt.savefig('images/violinplot_{}'.format(col))

    # Bar graphs of individual columns
    for col in categorical_vals:
        if col != 'last_trip_date':
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111)
            ax.set_title(col)
            sns.countplot(y=df[col], ax=ax, palette='pastel')
            plt.savefig('images/bargraph_{}'.format(col))

    # Checking for correlated features
    c = df.corr().abs()
    s = c.unstack()
    s.sort_values(ascending=False)

    # Dropping those correlated features
    df.drop(['sale_duration2', 'sale_duration', 'num_payouts', 'gts'], inplace=True, axis=1)

    # Temp csv for model testing use
    test_df = df.drop(['description', 'name',
                       'org_desc', 'org_name', 'payee_name', 'venue_name', 'venue_state', 'venue_country',
                       'country'], axis=1)
    test_df = pd.get_dummies(test_df, drop_first=True)
    test_df.reset_index()
    test_df.head()

    # Creating CSV for model testing
    test_df.to_csv('data/2000sample_alltextdatesdropped.csv')
