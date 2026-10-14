import pandas as pd
import matplotlib.pyplot as plt

import re
import unicodedata

import numpy as np


def remove_accents(text):
    text = unicodedata.normalize('NFKD', text)
    return ''.join(char for char in text if not unicodedata.combining(char))

def normalize_title_column(df):
    df = df.copy()
    
    df['title'] = (df['title']
                   .map(remove_accents)
                   .str.lower())
    
    df['title'] = (df['title']
                   .str.replace(r'\d{2}\s*mm', '', regex=True))
    
    return df


def remove_extreme_prices(df, max_price=2000):
    
    df = df.copy()
    
    df = df[(df['price'] <= max_price)]
    
    return df


def standardize_column_names(df):
    df = df.copy()
    
    df = df.rename(
        columns={'Case_Size_mm':'case_size', 'Seller_ID': 'seller_id', 'Is_Worldwide_Shipping':'worldwide_shipping'})
    
    return df

def standardize_condition(df):
    df = df.copy()
    
    condition_map= {
    'New':'New',
    'Nuevo':'New',
    'New other':'New',
    'Open box':'Open box',
    'Caja abierta':'Open box',
    'Used':'Used',
    'Usado':'Used',
    '--':'Unknown',
    'For parts or not working':'For parts or not working',
      'Excellent - Refurbished':'Excellent - Refurbished', 
       'Very Good - Refurbished':'Very Good - Refurbished', 
    'Good - Refurbished':'Good - Refurbished',
       'Certified - Refurbished':'Certified - Refurbished'
}
    df['condition'] = df['condition'].map(condition_map)
    
    # ----- Use titles to detect mislabeled condition -----
    mislabled_marker = (
    df['title']
    .str.contains(
        r'brand\s*new|unopened',
        case=False,
        regex=True,
        na=False
    )
    )

    df.loc[
        mislabled_marker &
        ~df['condition'].isin(['New', 'Open box']),
        'condition'
    ] = 'New'
    
    
    return df

def standardize_country(df):
    df = df.copy()
    
    country_map = {
    'Corea del Sur':'South Korea',
    'Reino Unido':'United Kingdom',
    'Estados Unidos':'United States'
    }

    df['country'] = df['Country'].map(country_map).fillna(df['Country'])
    
    return df

def drop_non_apple_brands(df):
    df = df.copy()
    
    df = df[df['brand']=='Apple']
    
    return df

def remove_bundles(df): # Also removes iphones
    df = df.copy()
    
    bundles = (df[df['title']
                  .str.contains(r'\bbundle\b|lot', regex=True, na=False)])

    df = (df[
        ~df['title']
             .str.contains(r'\bbundle\b|lot|iphone', regex=True, na=False)])
    
    return df

def drop_old_cols(df):
    df = df.copy()
    
    df = df.drop(columns = ['brand', 'Country'], errors='ignore')
    
    return df

def clean_dataframe(df):
    df = df.copy()
    
    df = standardize_country(df)
    df = standardize_condition(df)
    df = standardize_column_names(df)
    df = remove_extreme_prices(df)
    df = normalize_title_column(df)
    df = drop_non_apple_brands(df)
    df = remove_bundles(df)
    df = drop_old_cols(df)
    
    return df