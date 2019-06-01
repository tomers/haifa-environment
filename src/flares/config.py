BAZAN = 'בז"ן'
CAOL = 'כאו"ל'

use_local_files = True

if use_local_files:
    url = 'data/Flares.xls.html'
else:
    url = 'https://s3.amazonaws.com/haifa-nitur-public/Report2018/2017_2018_lapidim.xlsx'


factory_nitur_params = {
    BAZAN: {
        'url': url,
        'allowed_total_rates': [(850, 'H'), (650, 'Y')],
        'allowed_flare_rates': {
            'HHPFlare': None,
            'NEWFF': None,
            'OLDFF': None
        }
    },
    CAOL: {
        'url': url,
        'allowed_total_rates': (450, 'M'),
        'allowed_flare_rates': {
            'Flare-PP-185': (165, 'M'),
            'Flare-PP-180': (15, 'M'),
            'Flare-Monomers': (300, 'M')
        }
    }
}
