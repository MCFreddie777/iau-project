import numpy as np
import pandas as pd


def load(path):
	"""
	Loads dataset from csv file and omits its first column and sets it as a index
	:param path:
	:return:
	"""
	dataset = pd.read_csv(path, sep=",", index_col=0)
	dataset.index.name = None
	return dataset


def sanitize_boolean(boolean):
	"""
	Function takes string which resembles boolean value and returns boolean instead
	:param boolean: string
	:return: boolean representation of string in case string matches boolean pattern else returns None
	"""
	try:
		if boolean.strip() in ['f','F','FALSE','false','False']:
			return False
		elif boolean.strip() in ['t','T','TRUE','true','True']:
			return True
		else:
			return boolean #TODO: return None
	except AttributeError:
		return None

def sanitize_age(age):
	"""
	Takes age attribute and tries to parse string number into integer
	If fails, returns NaN
	:param age:
	:return:
	"""
	try:
		sanitized= int(pd.to_numeric(age,errors="coerce"))
		return sanitized if sanitized > 0 else np.nan
	except AttributeError:
		return np.nan
	except ValueError:
		return np.nan

def parse_personal_info(personal_info):
	"""
	Takes specific string, and splits and array of substrings by delimiters
	:param personal_info:
	:return:
	"""
	try:
		return np.array(personal_info.replace(' -- ', ',').replace('|', ',').replace('\r\r\n', ',').split(','))
	except AttributeError:
		return None

def remove_empty(arr):
	"""
	Replaces empty values (?,??) from array with None
	:param arr:
	:return:
	"""
	if arr is not None:
		empty = ['?', '??']
		for i in range(0, len(arr)):
			if (arr[i] != None):
				if arr[i] in empty:
					arr[i] = None
	return arr

def fill_value(number, arr):
	if arr is not None:
		return arr[number]
