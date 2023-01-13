# LINUX SHELL TUTORIAL

Nano text editor is a bit friendlier than Vim
  - ```nano myscript.sh```
<br><br><br>


## **Basic Commands**
  - ```cd, pwd```
  - ```ls -lha``` **long view, like windows dir**
    ```
        drwxrwxrwx  11 han  staff      352 Sep 16 19:25 somefolder
        -rwxrwxrwx   1 han  staff     4086 Oct  4 06:33 somefile.ipynb

        drwxrwxrwx (dir flag, user, usergroup, everyone)

        0 - nothing
        1 - execute
        2 - write
        3 - write/execute (2+1)
        4 - read
        5 - read/execute (4+1)
        6 - read/write (4+2)
        7 - read/write/execute (4+2+1)
    ```
  - ```cp, mv```
  - ```rm -r``` **for folder and everything in it, otherwise use rmdir**
  - ```echo``` **print**
  - ```cat``` **concatenate txt file to terminal**
    e.g.
    ```
        $ cat a.txt
        this is the contents
        of a.txt
    ```
  - ```less``` **similar to cat, but it launches a view. press Q to quit**
  - ```mkdir/rmdir, touch``` **make dir / make empty file**
  - ```chmod +rwx script.sh```
  - ```chmod 777 script.sh```
  - ```ls --help``` **help for most commands (guess mac doesn't have this)**
  - ```man + help``` **manual**
  - ```man ls```
  - **changing shell interpreter**
    - ```/bin/sh```
    - ```/bin/bash```
    - ```/bin/zsh```
  - ```grep``` **grabs data from a file or a command you are pursuing**
    - **e.g. is the verbose flag available for the mv command?**
    - ```mv --help | grep verbose```
  - ```df``` **list all volumes**

## **What is a shell**
- The shell is a command line interpreter. It translates commands entered by the user and converts them into a language that is understood by the kernel.

## **What is a shell script**
- a list of commands, listed in the order of execution.
- a good script includes comments preceded by ```#``` sign
- The initial shell type was the Bourne Shell, C Shell came later
  - Bourne Shell (and most common subtypes:)
    - Bourne
    - Korn
    - Bourne-Again
    - POSIX
  - C Shell (subtypes:)
    - C
    - TENEX/TOPS X
    - Z (popular recently)

### **Basic Shell Script**
- shebang points to interpreter path for the shell you wish to use.
- all script in linux execute using the interpreter specified in your first line.
- **Remember to chmod +x before executing**
```sh
      #!/bin/sh

      # Author : Han
      # Script is as follows

      echo "What is your name?"
      read PERSON
      echo "Hello, $PERSON"
```

--------------------------
## **Variables**

### **Variable Types**
- **Local**
  - present within current instance of shell.
  - not available to programs that are started by the shell
  - set at the command prompt
- **Environment**
  - available to any child process of the shell
  - some programs need env variables in order to function correctly
- **Shell (Global)**
  - special variable that is set by the shell and is required by the shell in order to function correctly
  - some of these are environment, some are local variables.

### **Special Variables**
- ```$0``` - filename of this script
- ```$1...9``` - script args
- ```$#``` - number of args applied to script
- ```$*``` - return all args that are double quoted
- ```$@``` - return individual args that are double quoted ($@ behaves like $* except that when quoted the arguments are broken up properly if there are spaces in them.)
- ```#?``` - exit status of last command that you executed
- ```$$``` - process ID of the current shell for the shell script.

##### - **_Exit Status \$?_**
  - run in terminal after previous script, to check exit status of the last command
  - typically 0 success, 1 failed, other ints if specific return states specified in the command
```
    $ echo $?
    0
```

##### - _**Differences between using quotes in args**_
```sh
        #!/bin/sh

        echo "Filename: $0"
        echo "1st arg: $1"
        echo "2nd arg: $2"
        echo "Quoted values: $@"
        echo "Quoted values: $*"
        echo "Num of args: $#"
```

| ./var.sh Han Haffidz          | ./var.sh "Han Haffidz"        | ./var.sh "Han" "Haffidz" |
|-----|-----|-----|
| Filename: ./var.sh <br> 1st arg: Han         <br> 2nd arg: Haffidz <br> Quoted values: Han Haffidz <br> Quoted values: Han Haffidz <br> Num of args: 2 | Filename: ./var.sh <br> 1st arg: Han Haffidz <br> 2nd arg:         <br> Quoted values: Han Haffidz <br> Quoted values: Han Haffidz <br> Num of args: 1 | Filename: ./var.sh <br> 1st arg: Han         <br> 2nd arg: Haffidz <br> Quoted values: Han Haffidz <br> Quoted values: Han Haffidz <br> Num of args: 2

##### - **_Differences between \$\* and \$@_**
<!-- ```
        | !/bin/sh              | !/bin/sh              | #!/bin/sh             | #!/bin.sh             | #!/bin/sh             |
        | for var in "$@"       | for var in $@         | for var in "$@"       | for var in $*         | for var in "$*"       |
        | do                    | do                    | do                    | do                    | do                    |
        |   echo "$var"         |   echo $var           |   echo $var           |   echo $var           |         echo $var     |
        | done                  | done                  | done                  | done                  | done                  |
        |                       |                       |                       |                       |                       |
        |-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|
        | $ ./var.sh a "b c" d  | $ ./var.sh a "b c" d  | $ ./var.sh a "b c" d  | $ ./var.sh a "b c" d  | $ ./var.sh a "b c" d  |
        | 1 a                   | a                     | a                     | a                     | a b c d               |
        | 2 b c                 | b                     | b c                   | b                     |                       |
        | 3 d                   | c                     | d                     | c                     |                       |
        |                       | d                     |                       | d                     |                       |
``` -->


```sh
#!/bin/sh            | OUTPUT:
for var in "$@"      | $ ./var.sh a "b c" d
do                   | a
  echo "$var"        | b c
done                 | d
                     |
---------------------|---------------------
#!/bin/sh            | OUTPUT:
for var in $@        | $ ./var.sh a "b c" d
do                   | a
  echo $var          | b
done                 | c
                     | d
---------------------|---------------------
#!/bin/sh            | OUTPUT:
for var in "$@"      | $ ./var.sh a "b c" d
do                   | a
  echo $var          | b c
done                 | d
                     |
---------------------|---------------------
#!/bin.sh            | OUTPUT:
for var in $*        | $ ./var.sh a "b c" d
do                   | a
  echo $var          | b
done                 | c
                     | d
---------------------|---------------------
#!/bin/sh            | OUTPUT:
for var in "$*"      | $ ./var.sh a "b c" d
do                   | a b c d
        echo $var    |
done                 |
                     |
```


## **Operators**
There are many variants amongst shells, we will cover the basic Bourne Shell here

### **Arithmetic Operators**
- ```+ - * / % = == !=```
- for conditional operators, conditions to be in square braces with spaces encapsulated.
  - e.g. ```[ $a == $b ]```
- all arithmetic done in long ints

### **Relational Operators**
- Won't work with strings
  - **usage:** ```[ $a -eq $b ]```
  - ```-eq``` (equal)
  - ```-ne``` (not equal)
  - ```-gt``` (greater than)
  - ```-lt``` (less than)
  - ```-ge``` (greater or equal)
  - ```-le``` (less or equal)


### **Boolean Operators**
- ```!``` - logical negation - ```[ ! false ]``` is true
- ```-o``` - logical OR - ```[ $a -lt 20 -o $b -gt 100 ]```
- ```-a``` - logical AND - ```[ $a -lt 20 -a $b -gt 100 ]```

### **String Operators**
- ```=``` - equal
- ```!=``` - not equal
- ```-z``` - is zero-length - ```[ -z $a ]```
- ```-n``` - is non-zero-length - ```[ -n $a ]```
- ```str``` - is not empty - ```[ $a ]```

### **File Test Operators**
- **usage:** ```[ -b $file ]```
- ```-b``` - is it a block special file
- ```-c``` - is it a character special file
- ```-d``` - is it a directory
- ```-f``` - is it an ordinary file
- ```-g``` - does file have its set group ID (SGID) set
- ```-k``` - does file have its sticky bit set
- ```-p``` - is file a named pipe
- ```-t``` - is file descriptor open and associated with a terminal
- ```-u``` - does file have its Set User ID (SUID) set
- ```-r``` - is file readable
- ```-w``` - is file writable
- ```-x``` - is file executable
- ```-s``` - is filesize greater than 0
- ```-e``` - does file exist (also true if dir exists)

## **Shell Loops**
- **For**
  ```sh
    #!/bin.sh
    for var in list
    do
      echo $var
    done
    #############
    for var in 0 1 2 3 4 5 6 7 8 9 # this works too
  ```
- **While**
  ```sh
    #!/bin/sh
    a=0
    while [ $a -lt 10 ]
    do
      echo $a
      a=`expr $a + 1`
    done
  ```
- **Until**
  ```sh
    #!/bin.sh
    a=0
    until [ $a -gt 9 ]
    do
      echo $a
      a=`expr $a + 1`
    done
  ```
- **Loop Control**
  - **break**
  ```sh
          #!/bin.sh
          a=0
          while [ $a -lt 10 ]
          do
            echo $a
            if [ $a -eq 5 ]
            then
              break
            fi
            a=`expr $a + 1`
          done
  ```
  - **continue**
  ```sh
          #!/bin.sh
          NUMS="1 2 3 4 5 6 7"     | # OUTPUT
                                   | odd
          for NUM in $NUMS         | even
          do                       | odd
            Q=`expr $NUM % 2`      | even
            if [ $Q -eq 0 ]        | odd
            then                   | even
              echo "Even"          | odd
              continue             |
            fi                     |
            echo "Odd"             |
          done                     |
  ```


## Shell Functions
```sh
    #!/bin/sh

    Hello() {                  |
        echo "hello"           |
    }                          |
                               | #output
    Hello                      | hello
                               |
    ------------------------------------------
    # PASSING PARAMS
                               |
    HelloArgs (){              |
        echo "hello $1 $2"     |
    }                          |
                               | #output
    HelloArgs Han Haff         | hello Han Haff
                               |
    ------------------------------------------
    # RETURN
                               |
    HelloArgs (){              |
        echo "hello $1 $2"     |
        return 10              |
    }                          |
                               | #output
    HelloArgs Han Haff         | hello Han Haff
    ret=$?                     | Return value: 10
    echo "Return value: $ret"  |


```


## Use Case Examples

- Ping Home Network
```sh
        #!/bin/bash

        is_alive_ping()
        {
            ping -c l $1 > /dev/null
            [ $? -eq 0 ] && echo Node with IP: $1 is up.
        }

        for i in 192.168.2.{1...255}
        do
            is_alive_ping $i & disown
        done
        exit

        ---------------------------
        (base) han@Hanafis-MacBook-Pro shell % ./isaliveping.sh
        Node with IP: 192.168.1.10 is up.
        Node with IP: 192.168.1.1 is up.
        Node with IP: 192.168.1.4 is up.
        Node with IP: 192.168.1.2 is up.
        Node with IP: 192.168.1.30 is up.
        Node with IP: 192.168.1.5 is up.
        Node with IP: 192.168.1.9 is up.
        Node with IP: 192.168.1.13 is up.
        Node with IP: 192.168.1.255 is up.
        Node with IP: 192.168.1.11 is up.
```
- using a CRON scheduler, run this script hourly, and email admin if a host is down
```sh
        #!/bin/bash

        for i in $@
        do
            ping -c 1 $i &> /dev/null
            if [ $? -ne 0 ]; then
                echo "`date`: ping failed, $i host is down!" | mail -s "$i host is down!" gammaraysky@gmail.com
            fi
        done

        ---------------------------
        (base) han@Hanafis-MacBook-Pro shell % ./checkhosts.sh google.com yahoo.com 192.168.255.255

```
- extended version of the above
```sh
#!/bin/bash

LOG=/tmp/mylog.log
SECONDS=60
EMAIL=gammaraysky@gmail.com

for i in $@; do
    echo "$i-UP!" > $LOG.$i
done

while true; do
    for i in $@; do
        ping -c 1 $i > /dev/null
        if [ $? - ne 0 ]; then
            STATUS=$(cat $LOG.$i)
                if [ $STATUS != "$i-DOWN!" ]; then
                    echo "`date`: ping failed, $i host is down!" |
                    mail -s "$i host is down." $EMAIL
                fi
            echo "$i-DOWN!" >$LOG.$i
        else
            STATUS=$(cat $LOG.$i)
                if [ $STATUS != "$i-UP!" ]; then
                    echo "`date`: ping OK, $i host is up!"
                fi
            echo "$i-UP!" >$LOG.$i
        fi
    done
sleep $SECONDS
done

```