from typing import List
from enum import Enum
import pandas as pd


class RateFrequency(Enum):
  HOURLY = 1
  MONTHLY = 2
  YEARLY = 3

  @classmethod
  def from_str(cls, label: str): # TODO: declare return type
    str_to_freq = {
        'H': RateFrequency.HOURLY,
        'M': RateFrequency.MONTHLY,
        'Y': RateFrequency.YEARLY
    }
    assert label in str_to_freq, f'Rate frequency {label} is not supported. Available rates: {str_to_freq.keys()}'
    return str_to_freq[label]


class MaxRate:
    def __init__(self, rate: int, frequency: RateFrequency):
        self.rate = rate
        self.frequency = frequency

        
    def __str__(self):
        return f'MaxRate {self.rate} {self.frequency}'

    @property
    def is_hourly(self) -> bool:
        return self.frequency == RateFrequency.HOURLY

    @property
    def is_monthly(self) -> bool:
        return self.frequency == RateFrequency.MONTHLY

    @property
    def is_yearly(self) -> bool:
        return self.frequency == RateFrequency.YEARLY

class MaxRateCollection:
    def __init__(self, rates: List[MaxRate]):
        self.rates = list(filter(lambda rate: rate, rates)) if rates else []

    @property
    def hourly(self) -> MaxRate:
        return next(filter(lambda max_rate: max_rate.is_hourly, self.rates), None)

    @property
    def monthly(self) -> MaxRate:
        return next(filter(lambda max_rate: max_rate.is_monthly, self.rates), None)

    @property
    def yearly(self) -> MaxRate:
        return next(filter(lambda max_rate: max_rate.is_yearly, self.rates), None)
