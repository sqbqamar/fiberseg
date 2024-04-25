# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:34:33 2024

@author: drsaq
"""

import pandas as pd

# Read the Excel file
df = pd.read_excel('your_file.xlsx')

conversion_value = 0.65    # You change according to your microscopic standard

# Multiply the values in the 'Length', 'Width', and 'Area' columns by conversion_value
df['Length'] = df['Length'] *  conversion_value
df['Width'] = df['Width'] * conversion_value
df['Area'] = df['Area'] * conversion_value

# Write the modified DataFrame back to the Excel file
df.to_excel('Actual_measurement.xlsx', index=False)