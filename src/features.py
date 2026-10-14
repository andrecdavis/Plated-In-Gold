import re

def extract_model(text):
    text = str(text).lower()
    
    # Model Ultra
    ultra_num_match = re.search(r'ultra\s*(\d)', text)
    ultra_match = re.search(r'\bultra\b', text)
    if ultra_num_match:
        return f'Ultra {ultra_num_match.group(1).strip()}'
    if ultra_match:
        return 'Ultra 1'
    
    # Model SE
    se_num_match = re.search(r'\bse\s*(\d+)', text)
    se_match = re.search(r'\bse\b',text)
    if se_num_match:
        return f'SE {se_num_match.group(1).strip()}'
    if se_match:
        return 'SE 1'
    
    # Model Series
    series_num_match = re.search(r'\bseries\s*(\d+)', text)
    series_match = re.search(r'\bseries\b', text)
    s_match = re.search(r'\bs(\d+)\b|gen\s*(\d+)', text)
    watch_match = re.search(r'watch\s*(\d+)', text)
    series_zero = re.search(r'sport|a1553|a1554|1st|gen\s*1', text)
    serie_match = re.search(r'serie\s*(\d+)|seri\s*(\d+)', text)
    gen_match = re.search(r'\bgen\b\s*(\d+)', text)
    if series_num_match:
        return f'Series {series_num_match.group(1).strip()}'
    if gen_match:
        return f'Series {gen_match.group(1).strip()}'
    if series_match:
        return 'Series 1'
    if s_match:
        return f'Series {s_match.group(1)}'
    if watch_match:
        return f'Series {watch_match.group(1)}'
    if serie_match:
        return f'Series {serie_match.group(1)}'
    if series_zero:
        return 'Series 0'
    return None

def extract_family(text):
    text = str(text).lower()
    
    # Ultra family 
    ultra_fam_match = re.search(r'\bultra\b', text)

    if ultra_fam_match:
        return f'Ultra'
    
    # Series family 
    series_fam_match = re.search(r'\bseries\b', text)

    if series_fam_match:
        return f'Series'
    
    # SE family 
    se_fam_match = re.search(r'\bse\b', text)

    if se_fam_match:
        return f'SE'


def add_model_col(df):
    df = df.copy()
    
    df['model'] = df['title'].apply(extract_model)
    
    bad_models = ['None', 'Series 24', 'Series 44', 'Series None', 'SE 44']

    df = df[~((df['model'].isin(bad_models)) & (df['model'].notna())  )]
    
    return df


def add_family_col(df):
    df = df.copy()
    
    df['family'] = df['model'].apply(extract_family)
    
    return df



def add_premium_cols(df):
    df = df.copy()
    
    premium_patterns = {'hermes':r'\bhermes\b',
                    'cellular':r'cellular|lte',
                    'titanium':r'\btitanium\b',
                    'gold':r'24k|24ct|24\skarat',
                    'premium_band':r'milanese|link',
                    'ceramic':r'ceramic'
                   }
    
    for col, pattern in premium_patterns.items():
        
        df[col] = (df['title']
                    .str.contains(pattern, case=False, na=False)
                    .astype(int)
                  )
        
    # All Ultra models are titanium==1
    df.loc[
        df['model'].str.contains('ultra', case=False, na=False),
        'titanium'
    ] = 1
    
    return df




def add_features(df):
    df = df.copy()
    
    df = add_premium_cols(df)
    
    return df

def add_power_seller(df, min_listings=2):
    """1 if the seller appears in `min_listings`+ listings (notebook definition: 2+)."""
    df = df.copy()
    df['power_seller'] = (df['seller_id'].map(df['seller_id'].value_counts()) >= min_listings).astype(int)
    return df

def calculate_price_increase(df, feature):
    
    # Returns dollar_increase, percent_increase for gold vs retail models

    df_copy = df.copy()

    # Creates df of gold percentages

    plot_df = (
        df_copy.groupby('model')[feature]
        .mean()
        .mul(100)
        .sort_values()
        .to_frame('premium_pct')
    )

    # All models that have gold listings
    gold_list = [model for model in plot_df.index if (plot_df.loc[model, 'premium_pct'] > 0)]

    # Filter by gold models
    df_copy = df_copy[df_copy['model'].isin(gold_list)]

    df_comparison = (
        df_copy.groupby(['model', feature])['price']
        .mean()
        .unstack()
    )

    df_comparison = df_comparison.rename(columns = {0:'Retail', 1:feature })

    df_comparison['difference'] = df_comparison[feature] - df_comparison['Retail']
    df_comparison['percent'] = df_comparison[feature] / df_comparison['Retail']

    percent_increase = round((df_comparison['percent'].mean() - 1)*100, 1)
    dollar_increase = int(df_comparison['difference'].mean())

    return dollar_increase, percent_increase