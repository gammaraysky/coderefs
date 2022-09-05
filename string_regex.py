
####! STRING BASICS
len(text)
len(set([w.lower() for w in text4])) # .lower converts the string to lowercase.
[w for w in text2 if len(w) > 3] # Words that are greater than 3 letters long in text2
[w for w in text2 if w.istitle()] # Capitalized words in text2
[w for w in text2 if w.endswith('s')] # Words in text2 that end in 's'

text6 = text5.split(' ')
[w for w in text6 if w.startswith('#')]

##! BOOL CHECKS
t.isupper()
t.islower()
t.istitle()
t.isalpha()
t.isdigit()
t.isalnum()

t.startswith('0')
t.endswith('t')

##! CHANGE CASE
s.lower()
s.upper()
s.titlecase()

##! SPLIT, JOIN, SPLITLINES
a='ouagadougou'
print( [c for c in a] ) # ['o', 'u', 'a', 'g', 'a', 'd', 'o', 'u', 'g', 'o', 'u']
b=a.split('ou') # ['', 'agad', 'g', '']
c='ou'.join(b) # ouagadougou

t = 'To be or not to be is the question '
words = t.strip().split(' ') # strips left and right whitespace

text="line1\nline2\nline3"
print(  text.splitlines()  )

##! STRIP, RSTRIP, LSTRIP
# s.strip(); s.rstrip; s.lstrip # removes all forms of whitespace from front and back only.
# whitespace includes \n, DOS new lines ^M that shows up as \r or \r\n
t = ' To   be  or not to   be  '
print( t.split(' ') )
print( t.strip().split(' ') )
print( [w.lower() for w in t.split(' ') if len(w)>0] ) # unique set of words

##! FIND, RFIND  # find from the front, find from the back
##! REPLACE(searchstring, replacementstring)
t = 'A quick brown fox jumped over the lazy dog.'
print( t.find('o') )
print( t.rfind('o') )
print( t.replace('o', 'O') ) # A quick brOwn fOx jumped Over the lazy dOg.



##! FILE HANDLING
f = open('data/filereading.txt', 'r')
firstline = f.readline() # reads line by line and moves read head
print(  firstline  )

nextline = f.readline()
print(  nextline  )

a = f.readlines() # reads all lines and moves read head
print(  a  )
print()

f.seek(1) # moves read head to character
b = f.read() # reads all characters
print(  b  )
print(  len(b)  , 'characters')
print(  len(b.splitlines()) , 'lines'  )
print(  len(b.split()) , 'words'  )
print(  b.split()  )


##! REGEX
"""
### REGEX META-CHARACTERS
    .       : wildcard, matches a single character
    ^       : start of a string
    $       : end of a string
    []      : matches one of the set of characters within []
    [a-z]   : matches one of the range of chars a,b...z
    [^abc]  : matches a character that is not a,b, or c
    a|b     : matches either a or b, where a and b are strings
    ()      : scoping for operators
    \       : escape character for special chars (\t \n \b)
    \b      : matches word boundary
    \d      : any digit, equivalent to [0-9]
    \D      : any non-digit, equivalent to [^0-9]
    \s      : any whitespace, equivalent to [ \t\n\r\f\v] (space, tab, newline)
    \S      : any non-whitespace, equivalent to [^ \t\n\r\f\v]
    \w      : any alphanumeric char, equivalent to [a-zA-Z0-9_]
    \W      : any non-alphanumeric char, equivalent to [^a-zA-Z0-9_]
    
### REGEX REPETITIONS
    *       : matches 0 or more occurrences
    +       : matches 1 or more occurrences
    ?       : matches 0 or 1 occurrences
    {n}     : exactly n repetitions, n>0
    {n,}    : at least n repetitions
    {,n}    : at most n repetitions
    {m,n}   : at least m and at most n repetitions

### REGEX SCOPING/CAPTURE GROUPS
    [0-9]([a-z]{2})[0-9]      : match on the whole expression, return me what's inside the () e.g. 'a'
    [0-9](?:[a-z]{2})[0-9]    : return me the whole matched expression
    (?P<time>[a-z])           : capture named group
"""
import re 
[w for w in text8 if re.search('@[A-Za-z0-9_]+', w)]

# Write code that would extract @callouts from the following tweet:
tweet = '"Ethics are built right into the ideals and objectives of the United Nations" #UNSG @ NY Society for Ethical Culture bit.ly/2guVelr @UN @UN_Women'
callouts = [w for w in tweet.split() if w.startswith('@')]
print(  callouts  )

print("\nSo startswith is insufficient. we don't want standalone '@'\nThis is where regex comes in:\n")

callouts = [w for w in tweet.split() if re.search('@[A-Za-z0-9_]+', w)]
print(  callouts  )

tweet = '"Ethics are built right into the ideals and objectives of the United Nations" #UNSG @ NY Society for Ethical Culture bit.ly/2guVelr @UN @UN_Women'
callouts = [w for w in tweet.split() if re.search('@[A-Za-z0-9_]+', w)]
print(  callouts  )
callouts = [w for w in tweet.split() if re.search('@\w+', w)]
print(  callouts  )

capital = 'ouagadougou'
print(  re.findall(r'[aeiou]', capital)  )
print(  re.findall(r'[^aeiou]', capital)  )

##! REGEX DATES
dates = '23-9-2002\n\
         23/09/2002\n\
         23/9/02\n\
         9/23/2002\n\
         23 Sep 2002\n\
         2 Sep 2002\n\
         23 September 2002\n\
         23 September 2002\n\
         Sep 23, 2002\n\
         September 23, 2002\n'

print(  re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', dates)  )

print(  re.findall(r'\d{2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{2,4}', dates)  )
# why did it return that? because scoping operator means, it did match the whole string, but only returned you the scope you said was important.

print(  re.findall(r'\d{2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{2,4}', dates)  )
# ?: says it's unimportant, i.e this is a scoping operator, but don't just return me the match here.

print(  re.findall(r'\d{2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z,]* \d{2,4}', dates)  )

print(  re.findall(r'(?:\d{1,2} )?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z,]* (?:\d{1,2}, )?\d{2,4}', dates)  )

### DATES REGEX ASSIGNMENT
def date_sorter():
    # Your code here
    import pandas as pd
    pd.set_option('display.max_colwidth', None)
    pd.set_option("display.precision", 0)

    doc = []
    # with open('dates.txt') as file:
    with open('data/dates.txt') as file:
        for line in file:
            doc.append(line)

    df = pd.Series(doc)
    df.head()

    #? Search Patterns
    pattern1 = r'(?P<mm>\d{1,2})[/-](?P<dd>\d{1,2})[/-](?P<year>\d{2,4})'
    pattern2 = r'(?P<dd>\d{1,2})?[\s,\.-]*(?P<month>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z,\.-]*[\s]*(?P<dd2>\d{1,2})?[,]*(?:[a-z]{0,2})[,]?[\s\.-]+(?P<year>\d{2,4})'
    pattern3 = r'(?P<mm>\d{1,2})[/-](?P<year>\d{2,4})'
    pattern4 = r'(?P<month>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z,\.-]*[\s]+(?P<year>\d{2,4})'
    pattern5 = r'(?P<year>\d{4})'

    p1 = df.str.extract(pattern1, expand=True)
    p2 = df.str.extract(pattern2, expand=True)
    p3 = df.str.extract(pattern3, expand=True)
    p4 = df.str.extract(pattern4, expand=True)
    p5 = df.str.extract(pattern5, expand=True)


    def convertmonth(a):
        if a['month']=='Jan':
            a['mm']=1
        elif a['month']=='Feb':
            a['mm']=2
        elif a['month']=='Mar':
            a['mm']=3
        elif a['month']=='Apr':
            a['mm']=4
        elif a['month']=='May':
            a['mm']=5
        elif a['month']=='Jun':
            a['mm']=6
        elif a['month']=='Jul':
            a['mm']=7
        elif a['month']=='Aug':
            a['mm']=8
        elif a['month']=='Sep':
            a['mm']=9
        elif a['month']=='Oct':
            a['mm']=10
        elif a['month']=='Nov':
            a['mm']=11
        elif a['month']=='Dec':
            a['mm']=12
        return a

    #? Collate back extracted results into main df_result, all as numeric values
    #? For spelled months, convert and return the month in digit format.
    df_result = pd.DataFrame(df, index=df.index, columns=['text'])

    p4 = p4.apply(convertmonth, axis=1)
    p2 = p2.apply(convertmonth, axis=1)

    df_result.loc[p5['year'].notnull(), 'year'] = p5.loc[p5['year'].notnull(), 'year'].astype('float64', copy=True)
    df_result.loc[p3['year'].notnull(), 'year'] = p3.loc[p3['year'].notnull(), 'year'].astype('float64', copy=True)
    df_result.loc[p3['mm'].notnull(),  'month'] = p3.loc[p3['mm'].notnull(),     'mm'].astype('float64', copy=True)
    df_result.loc[p4['year'].notnull(), 'year'] = p4.loc[p4['year'].notnull(), 'year'].astype('float64', copy=True)
    df_result.loc[p4['mm'].notnull(),  'month'] = p4.loc[p4['mm'].notnull(),     'mm'].astype('float64', copy=True)
    df_result.loc[p1['mm'].notnull(),  'month'] = p1.loc[p1['mm'].notnull(),     'mm'].astype('float64', copy=True)
    df_result.loc[p1['dd'].notnull(),    'day'] = p1.loc[p1['dd'].notnull(),     'dd'].astype('float64', copy=True)
    df_result.loc[p1['year'].notnull(), 'year'] = p1.loc[p1['year'].notnull(), 'year'].astype('float64', copy=True)
    df_result.loc[p2['year'].notnull(), 'year'] = p2.loc[p2['year'].notnull(), 'year'].astype('float64', copy=True)
    df_result.loc[p2['mm'].notnull(),  'month'] = p2.loc[p2['mm'].notnull(),     'mm'].astype('float64', copy=True)
    df_result.loc[p2['dd'].notnull(),    'day'] = p2.loc[p2['dd'].notnull(),     'dd'].astype('float64', copy=True)
    df_result.loc[p2['dd2'].notnull(),   'day'] = p2.loc[p2['dd2'].notnull(),   'dd2'].astype('float64', copy=True)
    

    #? Convert 2-digit years between 50-99 to 1950-1999
    df_result['year'] = df_result['year'].astype('float64')
    df_result.loc[(df_result['year']<100) & (df_result['year']>=50), 'year'] += 1900
    df_result.loc[(df_result['year']<2000) & (df_result['year']>=1950)]
    # df_result.loc[455, 'text']


    #? fill missing months and days
    df_result.loc[df_result['month'].isnull(), 'month'] = 1
    df_result.loc[df_result['day'].isnull(), 'day'] = 1


    #? Concat day month year columns into a date column, and sort table by date
    df_result['date'] = [ '/'.join([str(int(x)),  str(int(y)), str(int(z))]) for x, y, z in zip(df_result['day'], df_result['month'], df_result['year']) ]
    df_result['date'] = pd.to_datetime(df_result['date'], dayfirst=True)
    df_result.to_csv('saved.csv')
    df_result = df_result.sort_values(['date'])
    # df_result

    resultindex = pd.Series(df_result.index.values)
    return resultindex
date_sorter()


##! REGEX PANDAS AND NAMED GROUPS
import pandas as pd

time_sentences = ["Monday: The doctor's appointment is at 2:45pm.", 
                  "Tuesday: The dentist's appointment is at 11:30 am.",
                  "Wednesday: At 7:00pm, there is a basketball game!",
                  "Thursday: Be back home by 11:15 pm at the latest.",
                  "Friday: Take the train at 08:10 am, arrive at 09:00am."]

df = pd.DataFrame(time_sentences, columns=['text'])
df

print("\n### number of chars in each string:")
print(  df['text'].str.len()                    )

print("\n### number of tokens in each string:")
print(  df['text'].str.split().str.len()        )

print("\n### does string contain 'appointment':")
print(  df['text'].str.contains('appointment')  )

print("\n### count how many times a digit occurs in each string")
print(  df['text'].str.count(r'\d')             )

print("\n### find all digits")
print(  df['text'].str.findall(r'\d')           )

print("\n### find all hours and minutes")
print(  df['text'].str.findall(r'(\d?\d):(\d\d)')  )

print("\nreplace weekdays with \"???\":  ")
print(  df['text'].str.replace(r'\w+day\b', '???' , regex=True)  )

print("\nreplace weekdays with 3 letter abbrevations:  ")
print(  df['text'].str.replace('(\w+day\b)', lambda x: x.groups()[0][:3], regex=True)  )

print("\ncreate new columns from first match of extracted groups  ")
print(  df['text'].str.extract(r'(\d?\d):(\d\d)')  )

print("\nextract the entire time, the hours, the minutes, and the period  ")
print(     df['text'].str.extract(r'((\d?\d):(\d\d) ?([ap]m))')  )
# extract only took the first instance (see last line)
print(  df['text'].str.extractall(r'((\d?\d):(\d\d) ?([ap]m))')  )
# so you need extractall

print("\nextract the entire time, the hours, the minutes, and the period with group names  ")
print(  df['text'].str.extractall(r'(?P<time>(?P<hour>\d?\d):(?P<minute>\d\d) ?(?P<period>[ap]m))')  )


####! ASCII/NON-ASCII, UTF8, INTERNATIONALIZATION
# PYTHON 3 uses UTF8 by default so no issue
# using u'string' is formatting as unicode in python 2.
text = 'Résumé'
print(text)

