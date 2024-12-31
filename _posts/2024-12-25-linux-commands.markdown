---
layout: post
title:  "Linux Commands"
date:   2024-12-25 00:00:00 +0800
categories: main
---

Useful Linux commands. Covering `grep`, `tr`, `awk`, `sed`, `sort`, `comm`, `uniq`, `wc`, `find`, `curl`, `tar`, `tee`, `cut`, `paste`

### grep

### tr

Used for translating or deleting characters. Supports transformations such as uppercase to lowercase, squeezing repeating characters, deleting specific characters and basic find and replace. 

```
❯ cat location
United States
SINGAPORE
Malaysia
Papua New Guinea
South Korea
```

Translate lower to uppercase, vice versa
```
❯ cat location | tr '[:lower:]' '[:upper:]'
UNITED STATES
SINGAPORE
MALAYSIA
PAPUA NEW GUINEA
SOUTH KOREA
❯ cat location | tr '[:upper:]' '[:lower:]'
united states
singapore
malaysia
papua new guinea
south korea

❯ cat location | tr '[a-z]' '[A-Z]'
UNITED STATES
SINGAPORE
MALAYSIA
PAPUA NEW GUINEA
SOUTH KOREA
```

Translate whitespaces into tables. Use `[:space:]` if you want to match any whitespace character, including spaces, tabs, newlines, etc.

```
❯ cat location | tr ' ' '\t'
United	States
SINGAPORE
Malaysia
Papua	New	Guinea
South	Korea

❯ cat location | tr '[:space:]' '\t'
United	States	SINGAPORE	Malaysia	Papua	New	Guinea	South	Korea	%
```

Translate characters into other character
```
❯ cat location | tr 'SE' '*'
United *tates
*INGAPOR*
Malaysia
Papua New Guinea
*outh Korea
```

Remove repetitive characters using `-s`

```
❯ echo "Sentence    with  too many   spaces" | tr -s " "
Sentence with too many spaces
```

Delete specified characters using `-d`

```
❯ echo "Sentence    with  too many   spaces" | tr -s " " | tr -d e
Sntnc with too many spacs
```

Delete digits

```
❯ echo "My number is 1234 5588" | tr -d '[:digit:]'
My number is

❯ echo "My number is 1234 5588" | tr -d '0-9'
My number is
```

Use completement with `-c` to inverse the deleted characters 

```
❯ echo "My number is 1234 5588" | tr -cd '[:digit:]'
12345588%
```

### awk

### sed

### sort

### comm

### uniq

### wc

### find

### curl



### tar



### tee

Reads the standard input and writes to both standard output and one or more files. Command is named after the T-splitter used in plumbing. Breaks the output of a program so it can be displayed and saved in a file. 

```
❯ seq 5
1
2
3
4
5
❯ seq 5 | tee five.txt
1
2
3
4
5
❯ cat five.txt
1
2
3
4
5
```

### cut

Cutting out each sectins from each line of files and writting result to standard output. Can be used to cut parts of line by byte, character and field. 

```
❯ cat location
United States
Singapore
Malaysia
Papua New Guinea
South Korea
```

Using `-b (bytes)`, show first 3 characters
```
❯ cut -b 1,2,3 location
Uni
Sin
Mal
Pap
Sou
```

First few characters with ranges
```
❯ cut -b 1-3,5-7 location
Unied
Sinapo
Malysi
Papa N
Souh K
```

First x character onwards
```
❯ cut -b 3- location
ited States
ngapore
laysia
pua New Guinea
uth Korea
```

Can also use `-c (character)` to achieve the same result as `-b`
```
❯ cut -c 1,2,3 location
Uni
Sin
Mal
Pap
Sou
```

Split by whitespace and only take 1st item
```
❯ cut -d " " -f 1 location
United
Singapore
Malaysia
Papua
South
```

Can also use it to take the 2nd item. Will return 1st item if there is no 2nd item
```
❯ cut -d " " -f 2 location
States
Singapore
Malaysia
New
Korea
```


### paste

Use to join files horizontally (parallel merging) by outputting lines consisting of lines from each specified file, separated by tab delimited (by default). 

```
❯ cat alphabet
a
b
c
d
e

❯ cat number_string
one
two
three
four
five
```

Add `number` to display row number
```
❯ paste number alphabet number_string
1	a	one
2	b	two
3	c	three
4	d	four
5	e	five
```

Use `-d` to define delimiter
```
❯ paste -d "," number alphabet number_string
1,a,one
2,b,two
3,c,three
4,d,four
5,e,five
```

Transpose
```
❯ paste -s number alphabet number_string
1	2	3	4	5
a	b	c	d	e
one	two	three	four	five
```

Convert into 3 columns
```
❯ cat alphabet | paste - - -
a	b	c
d	e
```

Split by whitespace and only take first item. Pipe it to `paste` and add `number`
```
❯ cut -d " " -f 1 state | paste - number
Arunachal	1
Assam	2
Andhra	3
Bihar	4
Chhattisgrah	5
```