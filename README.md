# Fuzzy Controller (TDD)

This project is an extensible fuzzy inference system (FIS) that can be used to create fuzzy controllers. It is written in Python and uses the [skfuzzy](https://github.com/scikit-fuzzy/scikit-fuzzy) library for fuzzy logic operations.

## Usage

Currently, there only exists a Singelton Inference system. To use this, simply ```from fuzzycontroller.system.singleton import SingletonFIS```. This class can be used to create an inference system.

**Loading Data**

  - ```sis = SingletonFIS()```
  - ```sis.load_data('your_data.json')```

The JSON data should be in the same format as specified in /fuzzycontroller/system/tests/data.json.

## TDD

This project was undertook because (a) I find fuzzy logic interesting, and (b) I have been reading a book called 'Agile Techinical Practices Distilled' and wanted to test out some TDD approaches. If you find any improvements to my TDD approach let me know! I'd love to learn more about this.
