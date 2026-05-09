# agents/tools.py
# Analytics tools for delivery intelligence

import pandas as pd
import numpy as np


def traffic_analysis(df):
    """
    Analyze delivery time by traffic density.
    Returns: dict of traffic levels and average delivery times
    """
    result = df.groupby('road_traffic_density')['time_takenmin'].agg(['mean', 'count'])
    return {
        level: {'avg_time': round(row['mean'], 2), 'count': int(row['count'])}
        for level, row in result.iterrows()
    }


def weather_analysis(df):
    """
    Analyze delivery time by weather conditions.
    Returns: dict of weather conditions and average delivery times
    """
    result = df.groupby('weatherconditions')['time_takenmin'].agg(['mean', 'count'])
    return {
        cond: {'avg_time': round(row['mean'], 2), 'count': int(row['count'])}
        for cond, row in result.iterrows()
    }


def vehicle_analysis(df):
    """
    Compare delivery performance by vehicle type.
    Returns: dict of vehicle types and stats
    """
    result = df.groupby('type_of_vehicle')['time_takenmin'].agg(['mean', 'median', 'std', 'count'])
    return {
        vehicle: {
            'avg_time': round(row['mean'], 2),
            'median_time': round(row['median'], 2),
            'std_time': round(row['std'], 2),
            'count': int(row['count']),
        }
        for vehicle, row in result.iterrows()
    }


def city_analysis(df):
    """
    Analyze delivery performance by city.
    Returns: dict of cities and stats
    """
    result = df.groupby('city').agg({
        'time_takenmin': ['mean', 'count'],
        'distance_km': 'mean',
    })
    return {
        city: {
            'avg_time': round(result.loc[city, ('time_takenmin', 'mean')], 2),
            'avg_distance': round(result.loc[city, ('distance_km', 'mean')], 2),
            'count': int(result.loc[city, ('time_takenmin', 'count')]),
        }
        for city in result.index
    }


def peak_hour_analysis(df):
    """
    Compare delivery times during peak vs non-peak hours.
    Returns: dict with peak vs non-peak stats
    """
    result = df.groupby('is_peak_hour').agg({
        'time_takenmin': 'mean',
        'distance_km': 'mean',
    })
    output = {}
    for peak, row in result.iterrows():
        label = 'peak' if peak == 1 else 'non_peak'
        output[label] = {
            'avg_time': round(row['time_takenmin'], 2),
            'avg_distance': round(row['distance_km'], 2),
        }
    return output


def rating_analysis(df):
    """
    Analyze delivery time by driver ratings.
    Returns: dict of rating brackets and stats
    """
    bins = [0, 2.5, 3.5, 4.0, 4.5, 5.1]
    labels = ['1.0-2.5', '2.5-3.5', '3.5-4.0', '4.0-4.5', '4.5-5.0']
    df = df.copy()
    df['rating_bracket'] = pd.cut(df['delivery_person_ratings'], bins=bins, labels=labels, right=False)
    result = df.groupby('rating_bracket', observed=True)['time_takenmin'].agg(['mean', 'count'])
    return {
        str(bracket): {'avg_time': round(row['mean'], 2), 'count': int(row['count'])}
        for bracket, row in result.iterrows()
    }


def distance_analysis(df):
    """
    Analyze delivery time by distance quintiles.
    Returns: dict of distance ranges and avg times
    """
    df = df.copy()
    df['distance_quintile'] = pd.qcut(df['distance_km'], 5, labels=[
        '0-20%', '20-40%', '40-60%', '60-80%', '80-100%'
    ])
    result = df.groupby('distance_quintile', observed=True).agg({
        'time_takenmin': 'mean',
        'distance_km': ['min', 'max'],
    })
    return {
        str(q): {
            'avg_time': round(result.loc[q, ('time_takenmin', 'mean')], 2),
            'dist_range': f"{result.loc[q, ('distance_km', 'min')]:.1f} - {result.loc[q, ('distance_km', 'max')]:.1f} km",
        }
        for q in result.index
    }


def festival_analysis(df):
    """
    Analyze festival impact on delivery times.
    Returns: dict of festival status and stats
    """
    result = df.groupby('festival')['time_takenmin'].agg(['mean', 'count'])
    return {
        status: {'avg_time': round(row['mean'], 2), 'count': int(row['count'])}
        for status, row in result.iterrows()
    }


def general_stats(df):
    """
    Get general delivery statistics.
    Returns: dict of key statistics
    """
    return {
        'total_deliveries': len(df),
        'avg_delivery_time': round(df['time_takenmin'].mean(), 2),
        'median_delivery_time': round(df['time_takenmin'].median(), 2),
        'avg_distance': round(df['distance_km'].mean(), 2),
        'avg_pickup_delay': round(df['pickup_delay_min'].mean(), 2),
        'peak_hour_percentage': round((df['is_peak_hour'].sum() / len(df)) * 100, 1),
        'avg_rating': round(df['delivery_person_ratings'].mean(), 2),
        'cities': df['city'].nunique(),
        'vehicle_types': df['type_of_vehicle'].nunique(),
    }


def comprehensive_summary(df):
    """
    Build a comprehensive data summary for general/open-ended questions.
    Returns: dict with all analysis results
    """
    return {
        'general': general_stats(df),
        'traffic': traffic_analysis(df),
        'weather': weather_analysis(df),
        'vehicles': vehicle_analysis(df),
        'peak_hours': peak_hour_analysis(df),
        'ratings': rating_analysis(df),
    }