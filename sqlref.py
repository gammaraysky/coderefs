

#! 01	Select all rows from Table
SELECT * FROM [Table1]
print(df)

#! 02	Select Top 3 rows from Table
SELECT TOP 3 * FROM [Table1]
print(df.head(3))

#! 03	Select all rows and sort by Age in Ascending order (ensure null values appear first)
print(df.sort_values(by=['Age'], ascending=True, na_position='first'))
SELECT * FROM [Table1] ORDER BY Age DESC

#! 04	Select all rows and sort by Age in Descending order (ensure null values appear SELECT * FROM [Table1] ORDER BY Age
print(df.sort_values(by=['Age'], ascending=False, na_position='last'))
last)

#! 05	Select DISTINCT Column values
SELECT DISTINCT [Department] FROM [Table1]
print(df.Department.unique())

#! 07	Select count of rows
SELECT Count(*) FROM [Table1]
print(len(df))

#! 07	Select SUM of any column
SELECT sum(Salary) FROM [Table1]
print(df['Salary'].sum())

#! 08	Select all rows where specific column values are NULL
SELECT * FROM [Table1] WHERE [Age] IS NULL
print(df[df['Age'].isnull()])

#! 09	Select all rows where specific column values are NOT NULL
SELECT * FROM [Table1] WHERE [Age] IS NOT NULL
print(df[~df['Age'].isnull()])

#! 10	Select all rows where specific column values > numeric value i.e. 35
SELECT * FROM [Table1] WHERE [Age] > 35
print(df[df.Age >= 35])

#! 11	Select all rows where specific column values is conditioned by AND operator
SELECT * FROM [Table1] WHERE [Age] >=30 AND [Age] <= 40
print(df[(df.Age >= 30) & (df.Age <= 40)])

#! 12	Select all rows where specific column values is conditioned by OR operator
SELECT * FROM [Table1] WHERE [Department] ='HR' OR [Department] = 'Marketing'
print(df[(df.Department=='HR') | (df.Department=='Marketing')])

#! 13	Select all rows where specific column values is conditioned by NOT operator
SELECT * FROM [Table1] WHERE [Department] <> 'HR'
print(df[~(df.Department=='HR')])

#! 14	Group by the data by specific column (i.e. Department) and get sum of other column (i.e. Salary)
SELECT Department, SUM(Salary) AS 'Total' FROM [Table1] Group By Department
print(df.groupby(['Department'])['Salary'].sum())

#! 15	Replace NULL values in a specific column with the mean of the column
UPDATE [Table1] SET Age=(Select Avg(Age) From Table1) WHERE AGE IS NULL
SELECT * FROM [Table1]
print(df['Age'].fillna((df['Age'].mean()), inplace=True))
print(df)

#! 16	Replace NULL values in a specific column with the median of the column
UPDATE [Table1] SET Age=(
SELECT x.Age from Table1 x, Table1 y
GROUP BY x.Age
HAVING SUM(SIGN(1-SIGN(y.Age-x.Age))) = (COUNT(*)+1)/2)
WHERE AGE IS NULL
print(df['Age'].fillna((df['Age'].median()), inplace=True))
print(df)


#! 17	INNER Join
SELECT * FROM [Table1] T1 INNER JOIN [Table2] T2 ON T1.EmployeeID = T2.EmployeeID
print(pd.merge(df,df2,on='EmployeeID'))

#! 18	LEFT Join
SELECT * FROM [Table1] T1 LEFT JOIN [Table2] T2 ON T1.EmployeeID = T2.EmployeeID
print(pd.merge(df,df2,on='EmployeeID',how='left'))

#! 19	RIGHT Join
SELECT * FROM [Table1] T1 RIGHT JOIN [Table2] T2 ON T1.EmployeeID = T2.EmployeeID
print(pd.merge(df,df2,on='EmployeeID',how='right'))

#! 20	FULL OUTER Join
SELECT * FROM [Table1] T1 FULL OUTER JOIN [Table2] T2 ON T1.EmployeeID = T2.EmployeeID
print(pd.merge(df,df2,on='EmployeeID',how='outer'))

#! 21	INNER Join with multiple keys
SELECT * FROM [Table1] T1 INNER JOIN [Table2] T2 ON T1.EmployeeID = T2.EmployeeID AND T1.Name=T2.Name
print(pd.merge(df,df2,how='inner',left_on=['EmployeeID','Name'],right_on=['EmployeeID','Name']))

#! 22	RANK Function
SELECT *, RANK() OVER (ORDER BY Age DESC) FROM Table1
df[“AgeRank”] = df.Age.rank(ascending=False)
print(df)

#! 23	DENSE RANK Function
SELECT *, DENSE_RANK() OVER (ORDER BY Age DESC) FROM Table1
df[“AgeRank”] = df.Age.rank(method=”dense”, ascending=False)
print(df)

#! 24	ROW NUMBER Function
SELECT *, ROW_NUMBER() OVER (ORDER BY Age DESC) FROM Table1
df.insert(loc=0, column='RowNum', value=np.arange(len(df))+1)
print(df)

#! 25	CTE Function (CASE WHEN)SELECT Count(*) FROM [Table1]
SELECT *,
CASE WHEN Salary >0 AND Salary <=20000 THEN 'Low'
     WHEN Salary >20000 AND Salary <=40000 THEN 'Medium'
WHEN Salary >40000 THEN 'High'
ELSE 'None' END AS 'Category'
FROM Table1

df['Category'] = df.apply(lambda row:
    'High' if row['Salary'] > 40000
    else ('Medium' if row['Salary'] > 20000 and row['Salary'] <= 40000
    else ('Low' if row['Salary'] > 0 and row['Salary'] <= 20000
    else None)),axis = 1)
print(df)

	
