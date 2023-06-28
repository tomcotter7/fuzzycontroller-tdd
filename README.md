[![codecov](https://codecov.io/gh/tomcotter7/fuzzycontroller-tdd/branch/main/graph/badge.svg?token=YKUP2O26LK)](https://codecov.io/gh/tomcotter7/fuzzycontroller-tdd)

# Fuzzy Controller (TDD)

This project is an extensible fuzzy inference system (FIS) that can be used to create fuzzy controllers. It is written in Python and uses the [skfuzzy](https://github.com/scikit-fuzzy/scikit-fuzzy) library for fuzzy logic operations.

## Usage

There exists both a Singleton & Non-Singleton Version of the FIS. The singleton version can be used as follows:

  - ```from fuzzycontroller.system.singleton import SingletonFIS```
  - ```fis = SingletonFIS()```
  - ```sfis.load_data('your_data_file.json')```

The data should follow the format of `example.json`.

The Non-Singleton System can be used in a similar manner.

  - ```from fuzzycontroller.system.nonsingleton import NonSingletonFIS```
  - ```nsfis = NonSingleton()```
  - ```nsfis.load_date('your_data_file')```

The data file can be in the same format, as just the inputs will differ.

## TDD

This project was undertook because (a) I find fuzzy logic interesting, and (b) I have been reading a book called 'Agile Techinical Practices Distilled' and wanted to test out some TDD approaches. If you find any improvements to my TDD approach let me know! I'd love to learn more about this.
