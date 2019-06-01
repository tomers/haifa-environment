from typing import Tuple, Union
import datetime
import os

from xlrd import XLRDError
import pandas as pd


def load_df(url: str, sheet_name: Union[int, str] = 0) -> Tuple[pd.DataFrame, bool]:
    from_html = os.path.splitext(url)[1] in ['.htm', '.html']

    # Read from input file
    if from_html:
        try:
            sheets = pd.read_html(url, encoding='iso8859_8')  # TODO: get encoding as parameter
        except ValueError:
            print(f'Failed parsing {url}')
            raise
        assert sheets
        df = sheets[0]

        # Make the first row a column name, and drop it
        df.columns = df.iloc[0]
        df = df.reindex(df.index.drop(0))

        df.reset_index(inplace=True, drop=True)
    else:
        try:
            df = pd.read_excel(url, sheet_name=sheet_name)
        except XLRDError:
            print('Should be parsed as HTML?')
            raise

    assert not df.empty

    return df, from_html


def parse_input_df(
        df: pd.DataFrame,
        from_html: bool,
        num_header_rows: int,
        columns_name: str,
        drop_first_header: bool = False,
        num_last_rows_to_discard: int = None,
        num_columns_to_keep: int = None,
        column_translator: dict = None,
        convert_to_numeric: bool = True
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if num_columns_to_keep is not None:
        df.drop(df.columns[range(num_columns_to_keep, df.shape[1])], axis=1, inplace=True)
        assert not df.empty

    column_translator_internal = {
        'שעה': 'Time',
        'תאריך': 'Date'
    }
    if column_translator:
        column_translator_internal.update(column_translator)

    # Get headers and set it as dataframe columns
    df_headers = df.iloc[0:num_header_rows - 1].fillna('').transpose().reset_index(drop=True)
    assert not df_headers.empty, 'No headers'
    # Translate all Hebrew columns to English
    df_headers[0].replace(column_translator_internal, inplace=True)

    if drop_first_header:
        # Drop the first header, not before saving 'Date' and 'Time' header names
        # This is due to the header dataframe being in the following form:
        #            0               1
        #    0       Flares        HHPFlare
        #    1       Flares           NEWFF
        #    2       Flares           OLDFF
        #    3  CAOL Flares    Flare-PP-185
        #    4  CAOL Flares    Flare-PP-180
        #    5  CAOL Flares  Flare-Monomers
        #    6         Time
        #    7         Date
        df_headers[1] = df_headers.apply(lambda row: row[1] or row[0], axis=1)
        df_headers.drop(df_headers.columns[0], axis='columns', inplace=True)

    # Join multiple-line headers to a single line
    columns = df_headers.apply(lambda row: row.map(str).str.cat(sep=' ').strip(), axis=1)

    # Update dataframe with manipulated headers
    df.columns = columns
    df.columns.name = columns_name

    # Move units to a separate dataframe
    df_units = df.iloc[num_header_rows-1:num_header_rows].reset_index(drop=True)
    df_units.columns = columns

    df_units.drop(columns=['Date', 'Time'], axis=1, inplace=True)

    # Discard headers and units
    df.drop(df.head(num_header_rows).index, inplace=True)
    # Drop last garbage rows
    if num_last_rows_to_discard:
        df.drop(df.tail(num_last_rows_to_discard).index, inplace=True)

    # Fix bad input where midnight is '01/01/1900  0:00:00'
    # Convert the time to midnight, and increment day to the next day
    midnight_invalid = [datetime.datetime(1900, 1, 1, 0, 0, 0), '24:00']
    midnight_valid = datetime.time()

    for i in df[df['Time'].isin(midnight_invalid)].index:
        df.loc[i, 'Time'] = midnight_valid
        df.loc[i, 'Date'] = pd.to_datetime(df.loc[i, 'Date'], dayfirst=True) + datetime.timedelta(days=1)
    df.to_csv('after_fix_midnight.csv')

    # Make sure that Date and Time contain datetime values
    # (it is expected to be string when using read_html instead of read_excel)
    # TODO: make sure this does not corrupt dataframe read using read_html
    if from_html:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df.to_csv('after_to_datetime.csv')

        def normalize_time(x):
            if isinstance(x, str):
                return pd.Timestamp(x).to_pydatetime().time()
            return x  # TODO: consider converting pd.Timestamp to datetime.time
            # elif isinstance(x, pd.Timestamp):
            #     return x
            # else:
            #     return x

        df['Time'] = df['Time'].apply(normalize_time)

    # Create combined 'DateTime' with both date and time
    df['DateTime'] = df.apply(lambda x: datetime.datetime.combine(x['Date'].date(), x['Time']), axis=1)
    df.to_csv('after_combine.csv')
    # Create a DatetimeIndex and assign it to the dataframe.
    df.index = pd.DatetimeIndex(df['DateTime'])
    df.index.name = 'Hour'

    # Drop Date and Time columns, as they are rather redundant
    df.drop(columns=['Date', 'Time', 'DateTime'], axis=1, inplace=True)

    # Drop empty rows (usually as a result of missing data on future dates)
    df.dropna(how='all', inplace=True)

    if convert_to_numeric:
        df = df.apply(pd.to_numeric, errors='coerce')

    return df, df_units


def read_cache(url: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(url)
    except OSError:
        return pd.DataFrame() # File/URL not found


def units_df_to_dict(df_units: pd.DataFrame) -> dict:
    return {name: df_units[name].values[0] for name in df_units}

def read_nitur_excel(url: str,
                     num_header_rows: int,
                     columns_name: str,
                     sheet_name: Union[int, str] = 0,
                     drop_first_header: bool = False,
                     num_last_rows_to_discard: int = None,
                     num_columns_to_keep: int = None,
                     column_translator: dict = None,
                     convert_to_numeric: bool = True
                     ) -> Tuple[pd.DataFrame, dict]:

    # Try reading from cache
    cache_files = [f'{url}.parquet', f'{url}.units.parquet']
    df_data, df_units = [read_cache(cache_file) for cache_file in cache_files]
    if not df_data.empty and not df_units.empty:
        units = units_df_to_dict(df_units)
        return df_data, units

    input_df, from_html = load_df(url, sheet_name)

    df_data, df_units = parse_input_df(
        input_df,
        from_html,
        columns_name=columns_name,
        drop_first_header=drop_first_header,
        num_last_rows_to_discard=num_last_rows_to_discard,
        num_header_rows=num_header_rows,
        num_columns_to_keep=num_columns_to_keep,
        column_translator=column_translator,
        convert_to_numeric=convert_to_numeric
    )

    # Store cache
    df_data.to_parquet(cache_files[0])
    df_units.to_parquet(cache_files[1])

    units = units_df_to_dict(df_units)
    return df_data, units


# def clean_up(df: pd.DataFrame):
#     """Clean status strings inline. Sets them to zero"""

#     # TODO: Convert to masked-array
#     # df = np.ma.masked_where(np.isnan(df), df) # TODO: select ['NoData', 'Down']

#     # Cleanup status data: replace status strings with empty value
#     # df.replace(['NoData', 'Down'], 0, inplace=True)
#     df_num = df.apply(pd.to_numeric, errors='coerce')

#     return df_num # np.ma.masked_where(np.isnan(df_num), df_num)

# # df_converted = df_input.copy()
# # clean_up(df_converted)


def convert_units_from_tghr_to_khr(df: pd.DataFrame, units: dict):
    """Converts all T/Hr values to Kg/Hr (inlines)"""
    # columns_to_convert = [col for col in df if units[col] == 'T/HR']
    # df[columns_to_convert] *= 1000 # convert from ton to kg
    # units = {k: 'Kg/Hr' if v in ['T/HR', 'Ton/Hr'] else v for k, v in units.items()}
    columns_to_convert = [name for name, unit in units.items() if unit in {'T/HR', 'Ton/Hr'}]

    for col in columns_to_convert:
        if col not in df:
            continue
        # print(f'col {col} {type(col)}:')
        # x = df[col]
        # print(x)
        # print(ma.getmaskarray(df[col]))

        df[col] *= 1000  # convert from ton to kg
        units[col] = 'Kg/Hr'

# units_converted = units_input.copy()
# convert_units_from_tghr_to_khr(df_converted, units_input)

# url = '/Users/toshalev/Downloads/2017_2018_lapidim.xlsx'
# sheet_name=0
# factory='bazan'
# df_input, units = read_status_excel_file(url=url, sheet_name=sheet_name, factory=factory)
# df = clean_up(df_input)
# print('df=')
# with pd.option_context('display.max_rows', None, 'display.max_columns', df.shape[1]):
#     print(df)
# print('====')
# convert_units_from_tghr_to_khr(df, units)
