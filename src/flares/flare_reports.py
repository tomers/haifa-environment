#!/usr/bin/env python
# coding: utf-8

from collections import namedtuple
from enum import Enum
from typing import Callable, Dict, Optional, Union
from typing import List, Tuple
import calendar
import os

from bidi.algorithm import get_display
from mpl_toolkits.axes_grid1 import make_axes_locatable
import calmap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common.nitur_excel_parser import convert_units_from_tghr_to_khr, read_nitur_excel
from ..common.nitur_util import MaxRateCollection, MaxRate, RateFrequency
from .config import factory_nitur_params

# # Plot graphs

# ## Configuration

colormap_negative_name = 'Reds'
colormap_positive_name = 'Greens'
colormap_neutral_name = 'Blues'

colormap_negative = plt.cm.get_cmap(colormap_negative_name)
colormap_positive = plt.cm.get_cmap(colormap_positive_name)
colormap_neutral = plt.cm.get_cmap(colormap_neutral_name)

color_strength = 0.6

color_negative = colormap_negative(color_strength)
color_positive = colormap_positive(color_strength)
color_neutral = colormap_neutral(color_strength)

color_status = 'Gray'
color_max_rate_line = 'Yellow'

TOTAL = 'Total'

NO_LAW_STR = '(לא קיים היתר!)'


def law_str_with_units_func(rate):
    return f'(היתר: {rate} ק"ג/שעה)'


def law_str_func(rate):
    return f'(היתר: {rate})'


def law_str_unbounded_func(rate):
    return f'(היתר: {rate}, סקלה לא חסומה)'

# Utility code


def is_skip_plot(plot_type: Dict[str, str], filter_run: Dict[str, Union[str, Callable]]):
    if filter_run:
        for kind, name in filter_run.items():
            if callable(name):
                # Convert function name to a string
                name = name.__name__
            if kind in plot_type and name != plot_type[kind]:
                # Mismatch to filter field which exist in plot type
                return True
    return False


def get_flare_title(flare: str, factory: str):
    return f'בסך לפידי {factory}' if flare == 'Total' else f'בלפיד {factory} {flare}'


def get_mean_by_year_month(df: pd.DataFrame):
    """Get a dataframe on monthly flow rate average by month and year"""
    df = df.copy()
    mean_by_month = df.sum(axis=1).resample('M').mean()

    # TODO: make this more panda'ic
    res = dict()
    for year, df_year in mean_by_month.groupby(mean_by_month.index.year):
        # TODO: use month_name and fix sorting alphabetically
        df_year.index = df_year.index.month
        res[year] = df_year.to_dict()

    df = pd.DataFrame(res).T
    df.index.name = 'Year'
    df.columns.name = 'Month'
    df.sort_index(ascending=False, inplace=True)

    return df


Plot = namedtuple('Plot', ['figure', 'title', 'dataframe'])  # , verbose=True)

# ## 1. Plot daily flow rate average

# In[31]:


def plot_daily_flow_rate_average(
        df: pd.DataFrame,
        factory: str,
        flare: str,
        max_rates: MaxRateCollection,
        bound_scale: bool,
        filter_run: dict = None,
        **kwargs
        ) -> Optional[Plot]:
    """Plot heat-map for flare's average hourly flow rate per-day"""

    if is_skip_plot(dict(type='plot_daily_flow_rate_average', factory=factory, flare=flare), filter_run):
        return None

    max_hourly_rate = max_rates.hourly if max_rates else None

    if bound_scale and not max_hourly_rate:
        # Bounded plot can only be made on hourly-limited flare
        return None

    # Build title
    flare_title = get_flare_title(flare, factory)
    title = f'ממוצע יומי של ספיקה שעתית {flare_title}'
    title += '\nביחידות ק"ג/שעה'
    if not max_hourly_rate:
        title += f' {NO_LAW_STR}'
    elif not bound_scale:
        title += f' {law_str_unbounded_func(max_hourly_rate.rate)}'
    else:
        title += f' {law_str_func(max_hourly_rate.rate)}'

    # Plot graph
    num_years = len(df.sum(axis=1).resample('Y').sum())  # there must be a more elegant way to do this!
    fig, axes = calmap.calendarplot(
        df[flare],
        how='mean',  # Hourly
        vmax=max_hourly_rate.rate if bound_scale else None,
        fillcolor='grey',
        linewidth=0,
        fig_kws=dict(figsize=(15, 2 + 3 * num_years)),
        cmap=kwargs.pop('cmap', plt.cm.get_cmap(colormap_negative_name)),
        **kwargs
    )
    fig.suptitle(get_display(title), fontsize=30)

    # Plot color bar
    fig.colorbar(axes[0].get_children()[1],
                 ax=axes.ravel().tolist(),
                 orientation='horizontal',
                 shrink=1.0,
                 extend='max' if bound_scale else 'neither'
                 )

    return Plot(fig, title, df)


# ## 2. Plot monthly flow rate average


def plot_monthly_flow_rate_average(
        df: pd.DataFrame,
        factory: str,
        flare: str = None,
        max_rates: Optional[MaxRateCollection] = None,
        filter_run: dict = None
        ) -> Optional[Plot]:
    """Plot heat-map for monthly flow rate average"""
    if is_skip_plot(dict(type='plot_monthly_flow_rate_average', factory=factory, flare=flare), filter_run):
        return None

    max_monthly_rate = max_rates.monthly if max_rates else None
    max_average_hourly_rate = max_monthly_rate.rate if max_monthly_rate else None

    # Build title
    flare_title = get_flare_title(flare, factory)
    title = f'ממוצע חודשי של ספיקה שעתית {flare_title}'
    title += '\nביחידות ק"ג/שעה'
    if max_average_hourly_rate:
        title += f' {law_str_func(max_average_hourly_rate)}'
    else:
        title += f' {NO_LAW_STR}'

    num_years = len(df.sum(axis=1).resample('Y').sum())  # there must be a more elegant way to do this!

    # Get data
    df = get_mean_by_year_month(df)

    # Plot graph
    fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=(15, 0 + 2 * num_years))
    axes = axes.T[0]
    ax = axes[0]

    # Configure plot
    im = ax.pcolor(df, cmap=colormap_negative)
    ax.set_aspect('equal')
    ax.set_title(get_display(title), fontsize=30)
    month_labels = [calendar.month_abbr[i] for i in df.columns]  # TODO: use more numpy'ic way to this
    month_ticks = range(len(month_labels))

    # Remove spines and ticks
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)

    # Plot months as x-labels
    ax.set_xlabel('')
    ax.xaxis.set_tick_params(which='both', length=0)
    ax.yaxis.set_tick_params(which='both', length=0)
    ax.set_xticks([i + 0.5 for i in month_ticks])
    ax.set_xticklabels([month_labels[i] for i in month_ticks], ha='center')

    # Plot year(s) as y-labels
    yticks_kws = dict(fontsize=32, color='grey', fontweight='bold', fontname='Arial', ha='center')
    yearticks = df.index
    ax.set_ylabel('')
    ax.set_yticks([i + 0.5 for i in range(len(yearticks))])
    ax.set_yticklabels(ax.get_yticklabels(), **yticks_kws)
    ax.set_yticklabels(yearticks, rotation='horizontal', va='center')
    ax.tick_params(axis='y', pad=50)

    # Plot each month's values in the middle of each cell
    for i, j in np.ndindex(df.shape):
        value = int(np.nan_to_num(df.iat[i, j]))
        is_exception = max_average_hourly_rate is not None and max_average_hourly_rate < value
        ax.text(j + 0.5,
                i + 0.5,
                value,
                ha='center',
                va='center',
                backgroundcolor=color_negative if is_exception else 'white',
                color='white' if is_exception else 'black',
                fontweight='bold' if is_exception else None)

    # Plot colorbar
    #   Create an axes on the right side of ax. The width of cax will be 5%
    #   of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.25)
    plt.colorbar(im, cax=cax)

    return Plot(fig, title, df)


# ## 3. Plot yearly flow rate average

# In[33]:


max_rate_line_kwargs = dict(color=color_max_rate_line, linestyle='dashed', linewidth=4)
max_rate_text_kwargs = dict(fontsize=16, horizontalalignment='center')
max_rate_bbox_kwargs = dict(facecolor='white', alpha=0.75)


def plot_yearly_flow_rate_average(
        df: pd.DataFrame,
        factory: str,
        flare: str = None,
        max_rates: Optional[MaxRateCollection] = None,
        filter_run: dict = None
        ) -> Optional[Plot]:
    """Plot yearly flow rate average per day"""
    if is_skip_plot(dict(type='plot_yearly_flow_rate_average', factory=factory, flare=flare), filter_run):
        return None

    max_yearly_rate = max_rates.yearly if max_rates else None

    df = df.copy()
    df = df.sum(axis=1).resample('Y').mean()  # get yearly mean
    df.index = df.index.year

    # Paint bars according to max rate
    def bar_color(value):
        if max_yearly_rate:
            return color_positive if value < max_yearly_rate.rate else color_negative
        else:
            return color_neutral
    colors = list(df.apply(bar_color))

    fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=(15, 8))
    axes = axes.T[0]
    ax = axes[0]
    ax = df.plot(ax=ax, kind="bar", color=colors)

    # Set plot title
    flare_title = get_flare_title(flare, factory)
    title = f'ממוצע שנתי של ספיקה שעתית {flare_title}'
    title += '\nביחידות ק"ג/שעה'
    if max_yearly_rate:
        title += f' {law_str_func(max_yearly_rate.rate)}'
    else:
        title += f' {NO_LAW_STR}'
    ax.set_title(get_display(title), fontsize=30)

    # Plot maximum rate line and label
    if max_yearly_rate:
        ax.axhline(y=max_yearly_rate.rate, **max_rate_line_kwargs)

        t = ax.text(np.mean(ax.get_xlim()),
                    max_yearly_rate.rate * 0.9,
                    get_display(f'היתר: {max_yearly_rate.rate}'),
                    verticalalignment='top',
                    **max_rate_text_kwargs)
        t.set_bbox(max_rate_bbox_kwargs)

    # Plot X and Y labels
    ax.xaxis.set_tick_params(labelsize=30)
    ax.set_xticklabels(ax.get_xticklabels(), rotation='horizontal')
    ax.set_xlabel(get_display('שנה'), fontsize=20)
    ax.set_ylabel(get_display('ממוצע פליטה שעתית (ק"ג/שעה)'), fontsize=20)

    # Plot exception as percentage above each bar
    if max_yearly_rate:
        for p in ax.patches:
            exception_percentage = 100.0 * (p.get_height() / max_yearly_rate.rate - 1.0)
            if 0 < exception_percentage:
                t = ax.annotate(get_display(f'{exception_percentage:.0f}% חריגה מההיתר'),
                                (p.get_x() + p.get_width() * 0.5, max_yearly_rate.rate * 0.7),  # p.get_height()
                                verticalalignment='top',
                                **max_rate_text_kwargs)
                t.set_bbox(max_rate_bbox_kwargs)

    return Plot(fig, title, df)


# 4. Plot daily number of exceptions from allowed hourly flow rate average

# In[]


def plot_daily_number_of_exceptions_from_allowed_hourly_flow_rate_average(
        df: pd.DataFrame,
        factory: str,
        max_rates: Optional[MaxRateCollection],
        filter_run: dict = None,
        **kwargs
        ) -> Optional[Plot]:
    """Plot heat-map for number of exceptions per-day"""
    if is_skip_plot(dict(type='plot_daily_number_of_exceptions_from_allowed_hourly_flow_rate_average',
                         factory=factory, flare=None), filter_run):
        return None

    if not max_rates:
        return None
    max_hourly_rate = max_rates.hourly
    if not max_hourly_rate:
        return None

    max_total_hourly_rate = max_hourly_rate.rate

    df = df.copy()
    df['Exception'] = df['Total'] > max_total_hourly_rate

    num_years = len(df.sum(axis=1).resample('Y').sum())  # there must be a more elegant way to do this!
    fig, axes = calmap.calendarplot(
        df['Exception'],
        fillcolor='grey',
        linewidth=0,
        fig_kws=dict(figsize=(15, 3 + 3 * num_years)),
        cmap=kwargs.pop('cmap', plt.cm.get_cmap(colormap_negative_name, 24)),
        **kwargs
    )

    cbar = fig.colorbar(axes[0].get_children()[1],
                        ax=axes.ravel().tolist(),
                        orientation='horizontal',
                        shrink=1.0,
                        ticks=[0, 6, 12, 18, 24])
    # cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_xlabel(get_display('שעות חריגה ביום'), fontsize=20)

    title = f'כמות חריגות יומיות מהיתר ספיקה שעתית בלפידי {factory}'
    title += '\nביחידות שעות חריגה ליום'
    fig.suptitle(get_display(title), fontsize=30)

    return Plot(fig, title, df)

# df_to_plot = df_exceptions.copy()
# plot_exceptions_heat_map(df_to_plot)
# None


# ## 5. Plot hourly flow rate average classification

# In[35]:


class FlareStatus(Enum):
    OK = 1
    NOK = 2
    STATUS = 3


def plot_hourly_flow_rate_average_histogram(
        ax,
        df: pd.DataFrame,
        flare: str,
        label_to_color: dict,
        max_rate: Optional[MaxRate]
        ):
    df_flare = df[flare]
    df_flare = df_flare[~np.isnan(df_flare)]
    _, _, patches = ax.hist(df_flare, bins=40, align='right')
    ax.set_xlabel(get_display('פליטה שעתית (ק"ג/שעה)'), fontsize=20)
    ax.set_ylabel(get_display('כמות שעות'), fontsize=20)

    # Remove top and right borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if max_rate:
        # Paint bars positive/negative
        for i, patch in enumerate(patches):
            # print(f'i={i}, patch={patch}')
            status = FlareStatus.OK if patch.xy[0] <= max_rate.rate else FlareStatus.NOK
            patch.set_facecolor(label_to_color[status])

        ax.axvline(max_rate.rate, **max_rate_line_kwargs)
        t = ax.text(max_rate.rate,
                    0.05 * ax.get_ylim()[1],
                    get_display(f'היתר: {max_rate.rate}'),
                    **max_rate_text_kwargs)
        t.set_bbox(max_rate_bbox_kwargs)


def plot_hourly_flow_rate_average_pie_chart(
        ax,
        df: pd.DataFrame,
        flare: str,
        label_to_heb: dict,
        label_to_color: dict,
        frequency_units_str: dict,
        max_rate: Optional[MaxRate]
        ):
    def calc_status(row: pd.Series) -> FlareStatus:
        total = row[flare]
        return FlareStatus.STATUS if np.isnan(total) else FlareStatus.OK if total <= max_rate.rate else FlareStatus.NOK

    df = df.copy()
    df['Status'] = df.apply(calc_status, axis=1)
    statuses = df['Status'].value_counts()

    status_labels = [label_to_heb[status] for status in statuses.index]
    colors = [label_to_color[status] for status in statuses.index]
    percents = 100.*statuses/statuses.sum()

    # TODO: just write 'years'
    unit_str = frequency_units_str[max_rate.frequency]

    patches, texts = ax.pie(statuses,
                            # labels=labels,
                            colors=colors,
                            startangle=90,
                            # autopct=autopct,
                            textprops=dict(size=16)
                            )

    # Set legend lavels (sorted by percent, descending)
    labels = [get_display(f'{status_label} - {percent:.1f}% ({absolute:d} {unit_str})')
              for percent, absolute, status_label in zip(percents, statuses, status_labels)]
    patches, labels, dummy = zip(*sorted(zip(patches, labels, statuses),
                                         key=lambda x: x[2],
                                         reverse=True))
    ax.legend(patches, labels, loc='lower center', fontsize=16)


def plot_hourly_flow_rate_average_classification(
        df: pd.DataFrame,
        factory: str,
        flare: str,
        max_rates: Optional[MaxRateCollection],
        filter_run: dict = None
        ) -> Optional[Plot]:
    """Hourly status pie chart"""

    plot_type = dict(type='plot_hourly_flow_rate_average_classification', factory=factory, flare=flare)
    if is_skip_plot(plot_type, filter_run):
        return None

    max_hourly_rate = max_rates.hourly if max_rates else None

    label_to_heb = {
        FlareStatus.OK: 'בגבולות ההיתר',
        FlareStatus.NOK: 'חריגה מההיתר',
        FlareStatus.STATUS: 'סטטוס תקלה'
    }
    label_to_color = {
        FlareStatus.OK: color_positive,
        FlareStatus.NOK: color_negative,
        FlareStatus.STATUS: color_status
    }
    frequency_units_str = {
        RateFrequency.HOURLY: 'שעות',
        RateFrequency.MONTHLY: 'חודשים',
        RateFrequency.YEARLY: 'שנים'
    }

    is_plot_pie_chart = max_hourly_rate is not None

    ncols = 2 if is_plot_pie_chart else 1
    fig, axes = plt.subplots(nrows=1, ncols=ncols, squeeze=False, figsize=(15, 9))
    axes0 = axes.T[0]
    ax = axes0[0]

    # Set title
    flare_title = get_flare_title(flare, factory)
    title = f'סיווג סטטוס פליטה שעתי לאורך התקופה {flare_title}'
    if max_hourly_rate:
        title += f'\n{law_str_with_units_func(max_hourly_rate.rate)}'
    else:
        title += f'\n{NO_LAW_STR}'
    fig.suptitle(get_display(title), fontsize=30)

    # Plot histogram
    plot_hourly_flow_rate_average_histogram(ax, df, flare, label_to_color, max_hourly_rate)

    # Plot pie chart
    if is_plot_pie_chart:
        axes1 = axes.T[1]
        ax = axes1[0]
        plot_hourly_flow_rate_average_pie_chart(ax, df, flare, label_to_heb, label_to_color, frequency_units_str,
                                                max_hourly_rate)

    return Plot(fig, title, df)


# ## Plot processing utility code


def allowed_total_rate_tuple_to_max_rate(
        allowed_total_rate: Union[Tuple[int, str], List[Tuple[int, str]]]
        ) -> Optional[MaxRateCollection]:

    if allowed_total_rate is None:
        return None

    allowed_total_rates = allowed_total_rate if isinstance(allowed_total_rate, list) else [allowed_total_rate]

    res = []
    for total_rate in allowed_total_rates:
        rate = total_rate[0]
        frequency_str = total_rate[1]
        frequency = RateFrequency.from_str(frequency_str)
        res.append(MaxRate(rate, frequency))

    return MaxRateCollection(res)


def save_plot(plot, factory: str, flare: str = None):
    export_dir = os.path.abspath(os.path.join('exports', factory))
    if flare and flare != TOTAL:
        export_dir = os.path.join(export_dir, flare)

    if not os.path.isdir(export_dir):
        os.makedirs(export_dir)

    title = ' '.join(plot.title.replace('/', '-').split('\n'))
    plot.figure.savefig(os.path.join(export_dir, f'{title}.png'))


def post_process_plot(plots, factory: str, flare: str = None):
    if not isinstance(plots, list):
        plots = [plots]

    for plot in plots:
        if plot is None:
            return
        # watermark_plot(plot)
        save_plot(plot, factory, flare)
        plt.close()


# ## Execute all flare plots

# In[38]:


def flare_plots(df: pd.DataFrame,
                factory: str,
                flare: str,
                max_rates: Optional[MaxRateCollection],
                filter_run: Optional[dict],
                **kwargs
                ):
    df_flare = pd.DataFrame(df[flare])

    def process(plot):
        post_process_plot(plot, factory, flare)

    # 1A. Plot daily flow rate average (unbounded, for non-hourly-limited flare)
    process(plot_daily_flow_rate_average(
        df_flare,
        factory,
        flare,
        max_rates,
        bound_scale=False,
        filter_run=filter_run,
        **kwargs))

    # 1B. Plot daily flow rate average (bounded, for an hourly-limited flare)
    process(plot_daily_flow_rate_average(
        df_flare,
        factory,
        flare,
        max_rates,
        bound_scale=True,
        filter_run=filter_run,
        **kwargs))

    # 2. Plot monthly flow rate average
    process(plot_monthly_flow_rate_average(
        df_flare,
        factory,
        flare,
        max_rates,
        filter_run))

    # 3. Plot yearly flow rate average
    process(plot_yearly_flow_rate_average(
        df_flare,
        factory,
        flare,
        max_rates,
        filter_run))

    # 4. Plot daily number of exceptions from allowed hourly flow rate average
    process(plot_daily_number_of_exceptions_from_allowed_hourly_flow_rate_average(
        df_flare,
        factory,
        max_rates,
        filter_run=filter_run))

    # 5. Plot hourly flow rate average classification
    process(plot_hourly_flow_rate_average_classification(
        df_flare,
        factory,
        flare,
        max_rates,
        filter_run=filter_run
        ))


# ## Run all on Excel sheet

# In[39]:


def read_flares_excel_file(
        url: str,
        **kwargs
        ) -> Tuple[pd.DataFrame, dict]:
    """Reads flare input Excel file (for a given sheet), and returns a DataFrame"""

    column_translator = {
    }
    df, units = read_nitur_excel(
        url=url,
        num_header_rows=3,
        drop_first_header=True,
        num_last_rows_to_discard=10,
        # num_columns_to_keep=5,
        columns_name='Flare',
        column_translator=column_translator,
        convert_to_numeric=True,
        **kwargs
        )
    return df, units


def run_all_on_sheet(
        url: str,
        factory: str,
        allowed_total_rates: Union[Tuple[int, str], List[Tuple[int, str]]] = None,
        allowed_flare_rates: Dict[str, Tuple[int, str]] = None,
        filter_run: dict = None,
        **kwargs
        ) -> pd.DataFrame:
    """"""
    df, units = read_flares_excel_file(url=url, **kwargs)

    # Keep only columns that are included in allowed_flare_rates
    df = df[list(allowed_flare_rates.keys())]

    convert_units_from_tghr_to_khr(df, units)
    df[TOTAL] = df.sum(axis=1)  # Add total column

    max_flare_rates = {}
    if allowed_flare_rates:
        for flare, allowed_flare_rate in allowed_flare_rates.items():
            max_flare_rates[flare] = allowed_total_rate_tuple_to_max_rate(allowed_flare_rate)
    max_flare_rates[TOTAL] = allowed_total_rate_tuple_to_max_rate(allowed_total_rates)

    for flare in df:
        max_rates = max_flare_rates.get(flare)
        flare_plots(df, factory, flare, max_rates, filter_run)

    return df


# # Run all Excel files

# In[40]:


def run_all(filter_run: dict = None) -> Dict[str, pd.DataFrame]:
    res = {}
    for factory, kargs in factory_nitur_params.items():
        if is_skip_plot(dict(factory=factory), filter_run):
            continue
        res[factory] = run_all_on_sheet(
            factory=factory,
            filter_run=filter_run,
            **kargs)

    return res


# In[42]:

def run():
    filter_run = {}
    # Comment this line to run all
    # flare='NEWFF'
    # type='plot_monthly_exceptions_heat_map'
    # filter_run.update(dict(factory=BAZAN, flare='NEWFF'))

    run_only = {}
    # run_only.update(dict(type='plot_monthly_flow_rate_average')) # , factory=CAOL))

    try:
        filter_run.update(run_only)
    except NameError:
        # run_only might not be defined
        pass

    run_all(filter_run)
