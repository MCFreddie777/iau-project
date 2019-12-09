import numpy as np
import pandas as pd
import matplotlib as mat
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as sm_stats
import statsmodels.stats.api as sms
import scipy.stats as stats
from scipy.stats import boxcox
import statistics
from statistics import mode
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression


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
	try:
		if boolean.strip() in ['f', 'F', 'FALSE', 'false', 'False']:
			return 0
		elif boolean.strip() in ['t', 'T', 'TRUE', 'true', 'True']:
			return 1
		else:
			return np.nan
	except AttributeError:
		return np.nan


def sanitize_number(number):
	try:
		sanitized = int(pd.to_numeric(number, errors="coerce"))
		return sanitized if sanitized > 0 else np.nan
	except AttributeError:
		return np.nan
	except ValueError:
		return np.nan


def replace_special_chars_with_comma(personal_info):
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


def sanitize_pregnancy(data, store):
	data.pregnant = data.pregnant.map(sanitize_boolean)
	data.loc[(data.sex == 1), 'pregnant'] = 0
	data.loc[(data.sex.isnull()) & (data.pregnant == 1), 'sex'] = 0
	return data


def sanitize_sex(data, store):
	data.sex = data.sex.map(lambda sex: 1 if sex.strip() == 'Male' else 0)
	return data


def get_measure_year(data, store):
	years = [];

	def store_measure_year(age, date):
		if (pd.notnull(age)):
			years.append(int(date.split('-')[0]) + age)

	data.apply(lambda x: store_measure_year(x.age, x.date_of_birth), axis=1)

	store['measure_year'] = pd.Series(years).mode();


def fill_null_age_by_date(age, date, store):
	if (date):
		return (store['measure_year'] - int(date.split('-')[0]))[0]
	return np.nanb


def fill_null_age_by_mean(data, store):
	unique_sexes = data.sex.unique()

	if 'mean' not in store:
		store['mean'] = {}
		for sex in unique_sexes:
			mean = data.loc[(data.age.notnull()) & (data.sex == sex)].age.mean()
			store['mean'][sex] = mean
	for sex in unique_sexes:
		data.loc[(data.sex == sex) & (data.age.isna()), 'age'] = store['mean'][sex]


def sanitize_age(data, store):
	data.age = data.age.map(sanitize_number)

	if 'measure_year' not in store:
		get_measure_year(data, store)

	data.loc[data.age.isnull(), 'age'] = data[data.age.isnull()].apply(
		lambda x: fill_null_age_by_date(x.age, x.date_of_birth, store), axis=1
	)

	data.loc[(data.age < 0), 'age'] = np.nan

	fill_null_age_by_mean(data, store)

	return data


def sanitize_personal_info(data, store):
	parsed_info = data.personal_info.map(replace_special_chars_with_comma).map(remove_empty)

	personal_info_columns = ['employment', 'country', 'relationship_info', 'employment_info', 'race']

	for index, name in enumerate(personal_info_columns):
		data[name] = parsed_info.map(lambda x: fill_value(index, x))

	data = data.drop('personal_info', axis=1)
	return data


def sanitize_age_of_birth(data, store):
	data.date_of_birth = data.date_of_birth.map(sanitize_date)
	return data


def sanitize_date(date):
	date = str(date).replace('/', '-')
	date = date[:10]
	date = date.split("-")

	if date[0] != 'nan':
		if len(date[0]) != 4:
			if len(date[2]) == 2 and int(date[0]) > 31:
				new_date = "19" + date[0] + "-" + date[1] + "-" + date[2]

			elif ((len(date[2]) == 2) and (int(date[0]) < 31) and (int(date[2]) > 31)):
				new_date = "19" + date[2] + "-" + date[1] + "-" + date[0]

			elif ((len(date[2]) == 2) and (int(date[0]) < 31) and (int(date[2]) < 31)):
				new_date = "20" + date[2] + "-" + date[1] + "-" + date[0]
			else:
				new_date = date[2] + "-" + date[1] + "-" + date[0]
			return new_date
	return '-'.join(date)


def sanitize_income(data, store):
	data.income = data.income.map(
		lambda income: 0 if str(income).strip() == '<=50K' else 1 if str(income).strip() == '>50K' else np.nan)
	return data


def strip_string(string):
	try:
		return str(string).strip()
	except AttributeError:
		return None


def sanitize_string(string):
	string = strip_string(string);
	if string in ['None', 'nan', '??', '?']:
		return None
	return string


def sanitize_multiword_string(string):
	try:
		return sanitize_string(string).replace('-', ' ').replace('_', ' ')
	except AttributeError:
		return None


def sanitize_relationship_info(relationship, relationship_info):
	if relationship_info is None and relationship in ['Wife', 'Husband']:
		return 'Married'

	if relationship_info is not None and 'Married' in relationship_info:
		return 'Married'

	return relationship_info


def sanitize_relationship(relationship, relationship_info):
	if relationship_info == 'Never married':
		return 'Unmarried'

	return relationship


def sanitize_string_attrs(data, store):
	for column in ['relationship', 'relationship_info', 'employment', 'employment_info']:
		data[column] = data[column].map(sanitize_multiword_string)

	data.relationship_info = data.apply(lambda row: sanitize_relationship_info(row.relationship, row.relationship_info),
										axis=1)

	data.loc[data.relationship.isnull(), 'relationship'] = data[data.relationship.isnull()].apply(
		lambda row: sanitize_relationship(row.relationship, row.relationship_info), axis=1)

	for column in ['country', 'education', 'race']:
		data[column] = data[column].map(sanitize_string)

	return data


def deduplicate(data, store):
	data = data.drop_duplicates(['name', 'address', 'date_of_birth'], keep="last")
	return data


def predicate_kurt_oxygen(data, store):
	if 'imp_median' not in store:
		store['imp_median'] = Imputer(missing_values='NaN', strategy='median')
		store['imp_median'].fit(data[['std_oxygen']])
		store['imp_median'].fit(data[['mean_oxygen']])

	data['std_oxygen'] = store['imp_median'].transform(data[['std_oxygen']])
	data['mean_oxygen'] = store['imp_median'].transform(data[['mean_oxygen']])

	y = data.kurtosis_oxygen.dropna()
	x = data[['std_oxygen', 'mean_oxygen']].head(y.count())

	if 'lm_kurt_oxygen' not in store:
		store['lm_kurt_oxygen'] = LinearRegression()
		store['lm_kurt_oxygen'].fit(x, y)

	missing_kurtosis_oxygen = data[['std_oxygen', 'mean_oxygen']].loc[data['kurtosis_oxygen'].isna()]
	predictions = store['lm_kurt_oxygen'].predict(missing_kurtosis_oxygen)

	data.loc[data.kurtosis_oxygen.isna(), 'kurtosis_oxygen'] = predictions

	return data


def get_education_num(data, education):
	temp = 1000
	for en in data['education-num'].loc[data.education == education]:
		if (en > 0) & (en < temp):
			temp = en
	return temp if temp != 1000 else 0


def fill_education_values(data, store):
	store['education'] = {}
	for education in data.education.unique():
		store['education'][education] = (get_education_num(data, education))

	data.loc[:, 'education-num'] = data.education.map(lambda x: store['education'][x])

	def fn(x):
		for key, value in store['education'].items():
			if value == x:
				return key

	data.loc[:, 'education'] = data['education-num'].map(fn)

	return data


def fill_employment_values(data, store):
	data.loc[data['employment'].isnull(), 'employment_info'] = None

	df = data[['education', 'education-num', 'age', 'hours-per-week', 'employment_info', 'employment']]
	chopped = df.dropna(axis=0, how="any")
	X = np.array(chopped[['education-num', 'age', 'hours-per-week']])
	Y = np.array(chopped[['employment_info', 'employment']])

	if 'knn' not in store:
		store['knn'] = KNeighborsClassifier(n_neighbors=5)
		store['knn'].fit(X, Y)

	data['age'] = data['age'].astype(int)

	def toint(x):
		try:
			return int(x)
		except ValueError:
			return np.nan

	data['education-num'] = data['education-num'].map(toint)

	if 'imp_median2' not in store:
		store['imp_median2'] = Imputer(missing_values='NaN', strategy='median');
		store['imp_median2'].fit(data[['hours-per-week']])

	data['hours-per-week'] = store['imp_median2'].transform(data[['hours-per-week']])

	to_be_filled = data[
		['education', 'education-num', 'age', 'hours-per-week', 'employment_info', 'employment']].loc[
		data['employment_info'].isnull() == True]

	to_be_filled = (to_be_filled.iloc[:, 1:4])
	to_be_filled = np.array(to_be_filled)

	predicted = store['knn'].predict(to_be_filled)
	df_predicted = pd.DataFrame(predicted)

	data.loc[data['employment_info'].isnull(), 'employment_info'] = df_predicted[0].values
	data.loc[data['employment'].isnull(), 'employment'] = df_predicted[1].values

	return data


def fill_skewness_glucose(data, store):
	df = data[['kurtosis_glucose', 'skewness_glucose']]
	df = df.dropna(axis=0, how="any")

	X = df.loc[:, 'kurtosis_glucose'].values.reshape(-1, 1)

	if 'lm_skewness_glucose' not in store:
		store['lm_skewness_glucose'] = LinearRegression()
		store['lm_skewness_glucose'].fit(X, df['skewness_glucose'])

	data_null = data['kurtosis_glucose'].loc[(data['skewness_glucose'].isna()) & (~data['kurtosis_glucose'].isna())]
	temp = store['lm_skewness_glucose'].predict(np.array(data_null).reshape(-1, 1))
	data.loc[(data['skewness_glucose'].isna() & ~(data['kurtosis_glucose'].isna())), 'skewness_glucose'] = temp
	data.loc[data['skewness_glucose'].isna(), 'skewness_glucose'] = data['skewness_glucose'].mean()

	return data


def fill_pregnancy(data, store):
	df = data[['education', 'education-num', 'age', 'hours-per-week', 'sex', 'pregnant']]
	chopped = df.dropna(axis=0, how="any")

	X = np.array(chopped[['education-num', 'age', 'hours-per-week', 'sex']])
	Y = np.array(chopped[['pregnant']])

	if 'knn_pregnancy' not in store:
		store['knn_pregnancy'] = KNeighborsClassifier(n_neighbors=5)
		store['knn_pregnancy'].fit(X, Y)

	to_be_filled = data[['education-num', 'age', 'hours-per-week', 'sex', 'pregnant']].loc[
		data['pregnant'].isnull() == True]
	to_be_filled = (to_be_filled[['education-num', 'age', 'hours-per-week', 'sex']])
	to_be_filled = np.array(to_be_filled)
	predicted = pd.DataFrame(store['knn_pregnancy'].predict(to_be_filled))

	data.loc[data['pregnant'].isnull(), 'pregnant'] = predicted[0].values

	return data


def fill_relationship_info(data, store):
	data.loc[data.relationship_info.isna(), 'relationship_info'] = 'Unknown'
	df = data[['education-num', 'sex', 'age', 'relationship_info', 'relationship']]
	chopped = df.dropna(axis=0, how="any")
	X = np.array(chopped[['education-num', 'sex', 'age']])
	Y = np.array(chopped[['relationship']])

	if 'knn_relationship_info' not in store:
		store['knn_relationship_info'] = KNeighborsClassifier(n_neighbors=5);
		store['knn_relationship_info'].fit(X, Y)

	to_be_filled = data[['education-num', 'sex', 'age', 'relationship_info', 'relationship']].loc[
		data['relationship'].isnull() == True]

	to_be_filled = (to_be_filled[['education-num', 'sex', 'age']])
	to_be_filled = np.array(to_be_filled)

	predicted = pd.DataFrame(store['knn_relationship_info'].predict(to_be_filled))
	data.loc[data['relationship'].isnull(), 'relationship'] = predicted[0].values

	return data


def fill_mean_or_most_freq(data, store):
	from sklearn.base import TransformerMixin

	class MyImputer_num(TransformerMixin):

		def __init__(self, missing_value=np.nan):
			self.missing_value = missing_value
			self.mean = 0

		def _get_mask(self, X, value_to_mask):
			if np.isnan(value_to_mask):
				return np.isnan(X);
			else:
				return np.equal(X, value_to_mask)

		def fit(self, X, y=None):
			mask = self._get_mask(X, self.missing_value)
			self.mean = np.mean(X[~mask])
			return self

		def transform(self, X):
			mask = self._get_mask(X, self.missing_value)
			X[mask] = self.mean

			return X

	categorical_columns = ['capital-loss', 'income', 'std_glucose', 'fnlwgt', 'kurtosis_glucose', 'skewness_oxygen',
						   'capital-gain', 'mean_glucose']

	if 'my_imp_num' not in store:
		store['my_imp_num'] = dict()
		for col in categorical_columns:
			store['my_imp_num'][col] = MyImputer_num()
			store['my_imp_num'][col].fit(data[col])

	for col in categorical_columns:
		data.loc[:, col] = store['my_imp_num'][col].transform(data.loc[:, col])

	class MyImputer_cat(TransformerMixin):

		def __init__(self):
			"""
			Impute missing values.
			"""

		def fit(self, X, y=None):
			self.fill = X.value_counts().index[0]

			return self

		def transform(self, X, y=None):
			return X.fillna(self.fill)

	categorical_columns = ['education', 'country', 'race']

	if 'my_imp_cat' not in store:
		store['my_imp_cat'] = dict()
		for col in categorical_columns:
			store['my_imp_cat'][col] = MyImputer_cat()
			store['my_imp_cat'][col].fit(data[col])

	for col in categorical_columns:
		data[col] = store['my_imp_cat'][col].transform(data[col])

	return data


def use_quantiles(df, column):
	new_df = df.copy(deep=True)
	skew_val = stats.skew(new_df[column])

	if ((skew_val < -2) or (skew_val > 2)):
		minimum = new_df[column].min()
		minimum = minimum + (-minimum - minimum)
		new_df[column] = np.log(new_df[column] + minimum)

	perc_95 = new_df[column].quantile(.95)
	perc_05 = new_df[column].quantile(.05)
	new_df.loc[new_df[column] < perc_05, column] = perc_05
	new_df.loc[new_df[column] > perc_95, column] = perc_95
	return new_df


def resolve_outliers(data, store):
	columns = ['kurtosis_oxygen', 'skewness_glucose', 'mean_glucose', 'mean_oxygen', 'std_oxygen', 'skewness_oxygen',
			   'kurtosis_glucose', 'fnlwgt', 'std_glucose']

	for col in columns:
		data = use_quantiles(data, col)

	return data
