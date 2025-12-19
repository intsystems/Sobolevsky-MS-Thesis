# Automatic Hierarchical Summarization of Scientific Texts

**Author:** Fedor Sobolevsky

**Scientific supervisor:** Ph. D. Konstantin Vorontsov

## Abstract 
In an age of exponential growth in the amount of available information in the world, the task of structuring and systematizing scientific knowledge, as well as increasing its accessibility, is becoming especially urgent. A structured organization of the main ideas and results from a scientific publication can speed up the process of gaining knowledge from it. Hierarchical summaries are one of the types of structured text representation that allows you to move from the main to the details when studying a topic. Since human processing of scientific texts in order to create a hierarchical summary takes a lot of time, it becomes necessary to develop automatic hierarchical summarization methods that are not inferior in quality to manual summarization. 

Large language models (LLMs) are a promising tool for solving this problem. This paper examines the ability of large language models to build hierarchical representations of scientific publication texts. The main method of assessing the quality of hierarchical summarization, as well as conventional summarization, is to assess the similarity with the reference summary created by an expert. Since there are currently no samples for the task of hierarchical summarization of scientific texts, a pre-selection of hierarchical summaries of a number of scientific articles is created for comparison with those generated automatically. Hierarchical summarization using LLMs is evaluated in comparison with summaries from this sample, taking into account various aspects of the similarity of hierarchical summaries, such as the structure and semantics of the summary.

The methods of comparing text hierarchies used so far are based on comparing them at the lexical level and, as shown in this work, poorly take into account their structure and semantics in relation to phrasing. In this regard, this paper also proposes a new method for comparing text trees - text tree editing distance (TTED), based on editing distance and semantic proximity estimation using language models. To assess the informativeness of the distance function between text trees as an aggregation of different aspects of their differences, special quality coefficients are introduced, reflecting the sensitivity of the similarity function to semantic and structural differences in text trees in relation to text paraphrasing at the vertices, and unbiased estimates for these coefficients are proposed for a random sample of text trees. Using these coefficients, extensive testing of the proposed metric and its modifications is conducted compared to a baseline used in previous works to compare text hierarchies. Testing shows that TTED indeed captures significant differences between text trees more accurately than the previously used method. A practical implementation of TTED is also provided for further usage. 

## Installation & Usage
All the code for this project can be found in the [`code`](https://github.com/intsystems/Sobolevsky-MS-Thesis/code) directory of this repository.

It is recommended to use a fresh virtual environment of choice. For example:
```
python -m venv tted
source tted/bin/activate # for Linux
tted/bin/activate/bat # for Windows
```

The code and required dependencies can be install with the following code (it is recommended to use a fresh virtual environment of choice):
```
git clone https://github.com/intsystems/Sobolevsky-MS-Thesis
cd ./Sobolevsky-MS-Thesis/code
pip install -r requirements.txt
```

All the experiments for TTED can be found in [`tted_tests.ipynb`](https://github.com/intsystems/Sobolevsky-MS-Thesis/code/tted_tests.ipynb). The source code for TTED can be found in the [`/tted`](https://github.com/intsystems/Sobolevsky-MS-Thesis/code/tted) subdirectory. The data and prompts used for the experiments are located in the [`/data`](https://github.com/intsystems/Sobolevsky-MS-Thesis/code/data) subdirectory.

## Publications
- F. Sobolevsky and K. Vorontsov, "Text Tree Edit Distance: A Language Model-Based Metric for Text Hierarchies," _2025 IEEE XVII International Scientific and Technical Conference on Actual Problems of Electronic Instrument Engineering (APEIE), Novosibirsk, Russian Federation, 2025, pp. 1-5, doi: 10.1109/APEIE66761.2025.11289395._

## Conference Talks
- Sobolevskii F. A., Vorontsov K. V. "Text Tree Edit Distance: Comparing Text Hierarchies Using Language Models" _X International Conference «Knowledge-Ontology-Theory» (KNOTH-2025)_
