## Dual Vigilance Hypersphere Adaptive Resonance Theory - Companion Python Code

### Citation Request:

If you make use of this code please cite the following paper:

> Islam Elnabarawy, Leonardo Enzo Brito da Silva and Donald C. Wunsch, "Dual Vigilance Hypersphere Adaptive Resonance Theory," in 2019 IEEE Symposium Series on Computational Intelligence, SSCI 2019.

and refer to this Github repository as

> Islam Elnabarawy, Leonardo Enzo Brito da Silva and Donald C. Wunsch, "Dual Vigilance Hypersphere Adaptive Resonance Theory," 2019. [Online]. <br/>Available: https://github.com/ACIL-Group/DVHA

### Datasets:

The data sets used in the experiments could not be included here due to copyright reasons. They are available at:

1. UCI machine learning repository: 
<br/>http://archive.ics.uci.edu/ml

2. Fundamental Clustering Problems Suite (FCPS): 
<br/>https://www.uni-marburg.de/fb12/arbeitsgruppen/datenbionik/data?language_sync=1

3. Datasets package: 
<br/>https://www.researchgate.net/publication/239525861_Datasets_package

4. Clustering basic benchmark: 
<br/>http://cs.uef.fi/sipu/datasets

### Installation

**Requires python 3.6 or higher.**

To install the prerequisites:
 
`pip install -r requilrements.txt`

### Usage

The python scripts in this repository are used for running the experiments 
associated with the paper and processing the output to aggregate the results.
They rely on the [NuART-Py project](https://github.com/ACIL-Group/NuART-Py)
for the implementation of the clustering algorithms themselves.

Each of the script files can be executed directly or as commands through 
the `run.py` command line interface (CLI).

```
$ python src/run.py
usage: run.py [-h] [--no-tracebacks]
              [--verbosity {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
              [--no-log-colors]
              {FA,DVFA,HA,DVHA,gather,gather_raw,reorder,help} ...
```

*The `help` command will show the options for each of the commands; e.g. `python src/run.py FA help`.*

### Software License

https://github.com/ACIL-Group/DVHA/blob/master/LICENSE

```
   Copyright 2019 Islam Elnabarawy

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
