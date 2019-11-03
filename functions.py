import numpy as np
import pandas as pd

# Preto si musíme prekonvertovať atribút age do číselnej hodnoty.
# Najprv zistíme, aké máme unikátne hodnoty v záznamoch.
# data.age.unique()
# Vidíme, že sa tam nachádzajú nečíselné i záporné hodnoty, ktoré musíme odstrániť

def sanitize_age(age):
    try:
        sanitized= int(pd.to_numeric(age,errors="coerce"))
        return sanitized
    except AttributeError:
        return np.nan
    except ValueError:
        return np.nan

# Usage:
# data.age = data.age.apply(sanitize_age)

def null_negative(value):
    if (value < 0):
        return np.nan
    else:
        return value

# Usage:
# data.age = data.age.apply(null_negative)
