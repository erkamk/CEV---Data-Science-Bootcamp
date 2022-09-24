"""
Please provide a similar table for your favorite 7 cities of Turkey that include their area and population.
24.09.2022
"""

import pandas as pd

dictionary = {'City': ["Antalya", "Giresun", "Bursa", "Burdur", "Eskişehir", "İzmir", "Trabzon"],
              'Area': [20177, 6934, 10882, 7175, 13925, 11891, 4628],
              'Population': [2619832, 429984, 3139744, 256898, 898369, 	4425789, 807903]}

df = pd.DataFrame(data = dictionary)