import matplotlib.pyplot as plt
from .stocker import Stocker
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import requests

import matplotlib
import matplotlib as mpl
matplotlib.use('Agg')
from matplotlib import style
from matplotlib import pyplot as plt
plt.style.use('ggplot')
sm, med, lg = 10, 15, 20
plt.rc('font', size = sm)         # controls default text sizes
plt.rc('axes', titlesize = med)   # fontsize of the axes title
plt.rc('axes', labelsize = med)   # fontsize of the x & y labels
plt.rc('xtick', labelsize = sm)   # fontsize of the tick labels
plt.rc('ytick', labelsize = sm)   # fontsize of the tick labels
plt.rc('legend', fontsize = sm)   # legend fontsize
plt.rc('figure', titlesize = lg)  # fontsize of the figure title
plt.rc('axes', linewidth=2)       # linewidth of plot lines
plt.rcParams['figure.figsize'] = [18, 10]
plt.rcParams['figure.dpi'] = 150


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


def web_stocker_run(stock_ticker):

   ## Instantiate a Stocker Object
    microsoft = Stocker(stock_ticker)
    stock_history = microsoft.stock
    st.write(stock_history.head())

   # # Data Exploration
    st.pyplot(microsoft.plot_stock())
    st.pyplot(microsoft.plot_stock(start_date = '2020-01-03', stats = ['Daily Change', 'Volume'], plot_type='pct'))

   # ## Potential Profit
    st.write('If we want to feel good about ourselves, we can pretend as if we had the fortune of mind to invest in Microsoft at the beginning with 100 shares. \
        We can then evaluate the potential profit we would have from those shares. You can also change the dates if you feel like trying to lose money!')
    # st.pyplot(microsoft.buy_and_hold(start_date='2000-01-05', nshares=100))
    # st.pyplot(microsoft.buy_and_hold(start_date='2020-01-05', nshares=100))

   # Trends and Patterns
    st.write("An additive model represents a time series as an overall trend and patterns on different time scales \
        (yearly, monthly, seasonally). While the overall direction of Microsoft is positive, it might happen to \
            decrease every Tuesday, which if true, we could use to our advantage in playing the stock market.\
                 The Prophet library, developed by Facebook provides simple implementations of additive models. \
        It also has advanced capabilities for those willing to dig into the code. The Stocker object does the tough work for us so we can use it \
            to just see the results. Another method allows us to create a prophet model and inspect the results. This method returns two objects,\
                model and data, which we need to save to plot the different trends and patterns.")

    model, model_data = microsoft.create_prophet_model(days=90)
    st.subheader('Forecast - 90 Days')
    st.pyplot(model.plot(model_data))

    model, model_data = microsoft.create_prophet_model(days=180)
    st.subheader('Forecast - 180 Days')
    st.pyplot(model.plot(model_data))

    model, model_data = microsoft.create_prophet_model(days=360)
    st.subheader('Forecast - 360 Days')
    st.pyplot(model.plot(model_data))
    
    st.pyplot(model.plot_components(model_data))
    microsoft.reset_plot()


    st.subheader('Seasonality')
    st.write(microsoft.weekly_seasonality)
    microsoft.weekly_seasonality = True
    st.write(microsoft.weekly_seasonality)

    model, model_data = microsoft.create_prophet_model(days=180)
    st.pyplot(model.plot(model_data))
    st.pyplot(model.plot_components(model_data))
    st.write('...')

    """
    We have added a weekly component into the data. 
    We can ignore the weekends because trading only occurs during the week 
    (prices do slightly change overnight because of after-market trading, 
    but the differences are small enough to not make affect our analysis). 
    There is therefore no trend during the week. 
    This is to be expected because on a short enough timescale, 
    the movements of the market are essentially random. 
    It is only be zooming out that we can see the overall trend. Even on a yearly basis, 
    there might not be many patterns that we can discern. 
    The message is clear: 
    playing the daily stock market should not make sense to a data scientist! 
    """

   # Turn off the weekly seasonality because it clearly did not work! 

    microsoft.weekly_seasonality=False

   # Changepoints
    """
    One of the most important concepts in a time-series is changepoints.
    These occur at the maximum value of the second derivative. 
    If that doesn't make much sense, they are times when the series goes from increasing to 
    decreasing or vice versa, or when the series goes from increasing slowly to increasing rapidly. 
    
    We can easily view the changepoints identified by the Prophet model with the following method.
    This lists the changepoints and displays them on top of the actual data for comparison.
    """

    microsoft.create_model()
    st.pyplot(microsoft.changepoint_prior_analysis())
    
    microsoft.create_model()
    microsoft.changepoint_prior_validation

    microsoft.create_model()
    st.pyplot(microsoft.changepoint_date_analysis())
    
    """
    Prophet only identifies changepoints in the first 80% of the data, 
    but it still gives us a good idea of where the most movement happens.
    It we wanted, we could look up news about Microsoft on those dates and try to corroborate with the changes.
    However, I would rather have that done automatically so I built it into Stocker.

    If we specify a search term in the call to changepoint_date_analysis,
    behind the scenes, Stocker will query the Google Search Trends api for that term.
    The method then displays the top related queries, the top rising queries, and provides a graph. 
    The graph is probably the most valuable part as it shows the frequency of the 
    search term and the changepoints on top of the actual data. 
    This allows us to try and corroborate the search term with either the changepoints or the share price.
    """

    microsoft.create_model()
    model, model_data = microsoft.create_prophet_model(days=360)
    st.pyplot(model.plot(model_data))
    
    st.pyplot(microsoft.evaluate_prediction())

    microsoft.create_model()
    st.pyplot(microsoft.predict_future(days=30))


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


    st.header('PyTrends Requests')

    import requests
    def get_symbol(symbol):
        url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)
        result = requests.get(url).json()
        for x in result['ResultSet']['Result']:
            if x['symbol'] == symbol:
                return x['name']
    company = get_symbol(stock_ticker)


    from pytrends.request import TrendReq
    pytrends = TrendReq(hl='en-US', tz=360)
    kw_list = [company]
    pytrends.build_payload(kw_list, cat=0, timeframe='today 5-y', geo='', gprop='')

    st.subheader('Interest By Region')
    st.write(pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=True, inc_geo_code=False))

    st.subheader('Related Topics')
    st.text(pytrends.related_topics())

    st.subheader('Related Queries')
    st.table(pytrends.related_queries())

    st.subheader('Top USA Trending Searches')
    st.write(pytrends.trending_searches(pn='united_states')) # trending searches in real time for United States

    st.subheader('Top Global Charts')
    st.write(pytrends.top_charts(2020, hl='en-US', tz=300, geo='GLOBAL'))

    st.subheader('Suggestions')
    st.table(pytrends.suggestions(keyword='football'))

    st.subheader('Categories')
    st.text(pytrends.categories())


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *