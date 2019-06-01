
# # Playground

# In[ ]:


# def foo(x):
#      index, plot = x
#      return f'{index}: {plot.title}'
# list(map(foo, enumerate(plots_with_data_by_factory[BAZAN].plots)))
# df = plots_with_data_by_factory[BAZAN].dataframe.copy()
# from copy import copy

# print(dir(plt.cm))
# palette = copy(plt.cm.YlOrBr)
# palette.set_over('
# plot_flares_average_hourly_flow_rate_heat_maps(df, BAZAN, 850, 'H', cmap=palette)


# In[ ]:


# df = plots_with_data_by_factory[BAZAN].dataframe.copy()
# plot_daily_exceptions_heat_map(df=df, factory=BAZAN, max_total_hourly_rate=850)
# None


# In[ ]:


# axes.annotate('ABC',xy=(250,250),rotation=45,fontproperties=prop,color='white')
def plot_bazan_flare():
    df = plots_with_data_by_factory[BAZAN]
    df = df.copy()
    TOTAL = 'Total'
    df[TOTAL] = df.sum(axis=1)
    max_rate = MaxRate(850, RateFrequency.HOURLY)
    plot = plot_flare_heat_maps(df, BAZAN, TOTAL, max_rate, True)
    return df, max_rate

# df, max_rate = plot_bazan_flare()

# df['Total'].hist(bins=20)
