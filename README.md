# student-performance-probabilistic-modeling


Data Processing (Data_processing_250a.ipynb):

1. Binary Variables:

'sex' col: 'F' -> 0, 'M' -> 1

'address' col: 'U' (urban) -> 0, 'R' (rural) -> 1

'famsize' col: 'LE3' (less or equal to 3) -> 0, 'GT3' (greater than 3) -> 1

'Pstatus' col: 'A' (apart) -> 0, 'T' (together) -> 1

'school' col: 'GP' -> 0, 'MS' -> 1

'yes'/'no' cols: ['schoolsup', 'famsup', 'paid', 'activities', 'nursery','higher', 'internet', 'romantic']: no -> 0, yes -> 1

2. Numerical Variables:

'age': 15-17 -> 0 (high school age), 18-22 -> 1 (college/uni age)

Grade ('G1', 'G2', 'G3'): Fail (0-9) -> 0, Pass (10-14) -> 1, Good (15-20) -> 2

'absences': None (0) -> 0, Low (1-5) -> 1, Medium (6-15) -> 2, High (16+) -> 3

3. Nominal Variables:
   
'Mjob' & 'Fjob' columns:
at_home: 0, health: 1, other: 2, services: 3, teacher: 4

'reason' column:
course: 0, home: 1, other: 2, reputation: 3

'guardian' column:
father: 0, mother: 1, other: 2

4. Variables kept as the original:

'Medu' & 'Fedu': 0 - none,  1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education

'traveltime': 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour

'studytime': 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours

'failures': n if 1<=n<3, else 4

'famrel': from 1 - very bad to 5 - excellent

'freetime': from 1 - very low to 5 - very high

'goout': from 1 - very low to 5 - very high

'Dalc': from 1 - very low to 5 - very high

'Walc': from 1 - very low to 5 - very high

'health': from 1 - very bad to 5 - very good

MLE part:

1. MLE_internal_factor_and_whole_data_Yiming.ipynb: implement the basic MLE model with internal factor dataset and the whole dataset, also implement the Laplace Smoothing

2. mle.py: implement the MLE model with external factor dataset

EM part:

1. EM_code.py: implement the EM algorithm
