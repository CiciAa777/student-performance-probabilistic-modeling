# student-performance-probabilistic-modeling

Data Processing:

1. Binary Variables:
'sex' col: 'F' -> 0, 'M' -> 1

'address' col: 'U' (urban) -> 0, 'R' (rural) -> 1
'famsize' col: 'LE3' (less or equal to 3) -> 0, 'GT3' (greater than 3) -> 1
'Pstatus' col: 'A' (apart) -> 0, 'T' (together) -> 1
'school' col: 'GP' -> 0, 'MS' -> 1
'yes'/'no' cols: ['schoolsup', 'famsup', 'paid', 'activities', 'nursery','higher', 'internet', 'romantic']: no -> 0, yes -> 1

3. Numerical Variables:
'age': 15-17 -> 0 (high school age), 18-22 -> 1 (college/uni age)
Grade ('G1', 'G2', 'G3'): Fail (0-9) -> 0, Pass (10-14) -> 1, Good (15-20) -> 2
'absences': None (0) -> 0, Low (1-5) -> 1, Medium (6-15) -> 2, High (16+) -> 3

4. Nominal Variables:
'Mjob' & 'Fjob' columns:
at_home: 0
health: 1
other: 2
services: 3
teacher: 4

'reason' column:
course: 0
home: 1
other: 2
reputation: 3

'guardian' column:
father: 0
mother: 1
other: 2
