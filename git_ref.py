

#* In Azure Cloud Shell, to double-check that Git is installed:
git --version


###? CONFIGURE GIT
#* To configure Git, you must define some global variables: user.name and user.email. Both are required for you to make commits.
git config --global user.name "<USER_NAME>"
git config --global user.email "<USER_EMAIL>"

#* Check that it worked
git config --list
# http.sslcapath=/etc/ssl/certs
# user.name=hanafi haffidz
# user.email=han@abc.com


###? SETUP YOUR REPOSITORY
#* Git works by checking for changes to files within a certain folder. We'll create a folder to serve as our working tree (project directory) and let Git know about it, so it can start tracking changes. We tell Git to start tracking changes by initializing a Git repository into that folder.
# Start by creating an empty folder for your project, and then initialize a Git repository inside it.
# Create a folder named Cats. This folder will be the project directory, also called the working tree. The project directory is where all files related to your project are stored. In this exercise, it's where your website and the files that create the website and its contents are stored.
mkdir Cats
cd Cats

#* Now, initialize your new repository and set the name of the default branch to main:
# If you're running Git version 2.28.0 or later, use the following commands:
git init --initial-branch=main
git init -b main
# For earlier versions of Git, use these commands:
git init
git checkout -b main

#* Check
git status
# On branch master
# No commits yet
# nothing to commit (create/copy files and use "git add" to track) 


#* Get Help
git --help
git <command> --help


git status
git add
git commit
git log


###? MODIFY FILES
touch index.html
#* touch updates a file's last-modified time if the file exists. If the file doesn't exist, Git creates an empty file with that file name.

git status
#* Git responds by informing you that nothing has been committed, but the directory does contain a new file:
# No commits yet
# Untracked files:
#   (use "git add <file>..." to include in what will be committed)
#     index.html
# nothing added to commit but untracked files present (use "git add" to track)

git add .
#* tells Git to index all the files in the current directory that have been added or modified.
git status
# On branch main
# Initial commit
# Changes to be committed:
#   (use "git rm --cached <file>..." to unstage)
#     new file:   index.html

git commit index.html -m "Create an empty index.html file"
#* -m flag in this command tells Git that you're providing a commit message.
#* There are many different ways to phrase commit messages, but a good guideline is to write the first line so that it says what the commit does to the tree. It's also common to capitalize the first letter, and to leave off the closing period to save space. Imagine that the first line of the message completes the sentence starting with "When pushed, this commit will..." (create an empty index.html file)
# [main (root-commit) 87e874c] Create an empty index.html file
#  1 file changed, 0 insertions(+), 0 deletions(-)
#  create mode 100644 index.html
git status
#* Follow up with a git status command and confirm that the working tree is clean—that is, the working tree contains no changes that haven't been committed.
git log
# commit 87e874c4aeeb3f9692ae5d9875235353708d7dd5
# Author: User Name <user-name@contoso.com>
# Date:   Fri Nov 15 20:47:05 2019 +0000
# Create an empty index.html file

#* Now go and edit index.html and add in some content. you can see that Git is aware of the changes you made.
git status

git commit -a -m "Add a heading to index.html"
#* note this time we didn't do git add, git commit -a adds all files in git's index that have been modified. But it won't add new files, you still need git add for that.


#* edit index.html some more and then check:
git diff
# --- a/index.html
# +++ b/index.html
# @@ -1 +1,6 @@
# -<html></html>
# \ No newline at end of file
# +<html>
# +<body>
# +    asdf
# +</body>
# +
# +</html>
# \ No newline at end of file
#* The output format is the same as the Unix diff command, and the Git command has many of the same options. A plus sign appears in front of lines that were added, and a minus sign indicates lines that were deleted.
#* The default is for git diff to compare the working tree to the index. In other words, it shows you all the changes that haven't been staged (added to Git's index) yet. To compare the working tree to the last commit, you can use git diff HEAD.
#* If the command doesn't return to the prompt after it executes, enter q to exit the diff view.

#* Next, commit the change. Instead of using the -a flag, you can explicitly name a file to be staged and committed if Git already has the file in the index (the commit command looks only for the existence of a file).
git commit -m "Add HTML boilerplate to index.html" index.html

git diff
#* This time, git diff produces no output because the working tree, index, and HEAD are all in agreement.

###? .GITIGNORE CERTAIN FILES
#* Let's say you edit some more. Some editors like sed/vims/emacs create backups index.html.bak/index.html~/index.html.~1~
#* You don't want git to recognise these files.
code .gitignore
#* make a .gitignore file and type in the following:
*.bak
*~
#* .gitignore is a very important file in the Git world because it prevents extraneous files from being submitted to version control. Boilerplate .gitignore files are available for popular programming environments and languages.

git add -A
git commit -m "Make small wording change; ignore editor backups"
#* This example uses the -A option with git add to add all untracked (and not ignored) files, and the files that have changed, to the files that are already under Git control.

###? DIRECTORY CHANGES AREN'T RECOGNISED
#* let's say you add a empty folder. git will not recognise any change. git only recognises changes to files, not directories
mkdir css
git status

#* Sometimes, especially in the initial stages of development, you want to have empty directories as placeholders. A common convention is to create an empty file, often called .git-keep, in a placeholder directory.
touch css/.git-keep
git add css

###? REPLACE/REMOVE FILES
rm CSS/.git-keep
cd CSS
code site.css #* add and save some css
git add .
git commit -m "Add a simple stylesheet"

#* Unlike most VCSes, Git records the contents of your files rather than the deltas (changes) between them. That's a large part of what makes committing, branching, and switching between branches so fast in Git. Other VCSes have to apply a list of changes to get between one version of a file and another. Git just unzips the other version.

###? LIST COMMITS
#* Now that you have a reasonable number of changes recorded, you can use git log to look at them. As with most Git commands, there are plenty of options to choose from. One of the most useful is --oneline.

git log
# commit ae3f99c45db2547e59d8fcd8a6723e81bbc03b70
# Author: User Name <user-name@contoso.com>
# Date:   Fri Nov 15 22:04:05 2019 +0000

#     Add a simple stylesheet

# commit 4d07803d7c706bb48c52fcf006ad50946a2a9503
# Author: User Name <user-name@contoso.com>
# Date:   Fri Nov 15 21:59:10 2019 +0000

#     Make small wording change; ignore editor backups

# ...
git log --oneline
# ae3f99c Add a simple stylesheet
# 4d07803 Make small wording change; ignore editor backups
# f827c71 Add HTML boilerplate to index.html
# 45535f0 Add a heading to index.html
# a69fe78 Create an empty index.html file

#* Another useful option is -nX, where X is a commit number: 1 for the latest commit, 2 for the one before that, and so on. 
git log --oneline -n2 # see last two commits


###? FIX SIMPLE MISTAKES
#* eg you made a mistake to your css. you could just edit save and commit it now, but you want it to reflect the original commit:
#* The --amend option to git commit lets you change history
git commit --amend --no-edit
#* The --no-edit option tells Git to make the change without changing the commit message. You can also use --amend to edit a commit message, to add files that were accidentally left out of the commit, or to remove files that were added by mistake.

#! The ability to change history is one of Git's most powerful features. You must use it carefully. In particular, it's a bad idea to change any commits that have been shared with another developer, or which were published in a shared repository, like GitHub.


###? RECOVER DELETED FILE: GIT CHECKOUT
#* Imagine that you made a change to a source code file that broke the entire project, so you want to revert to the previous version of that file. Or perhaps you accidentally deleted a file altogether. 
git checkout -- index.html
#* You can also check out a file from an earlier commit (typically, the head of another branch), but the default is to get the file from the index. The -- in the argument list serves to separate the commit from the list of file paths. It's not strictly needed in this case, but if you had a branch named <file_name> (perhaps because that's the name of the file being worked on in that branch), -- would prevent Git from getting confused.


###? RECOVER FILES: GIT RESET
#* You also can delete a file by using git rm. This command deletes the file on your disk, but it also has Git record the file deletion in the index.
#* So, if you ran this command:
git rm index.html
git checkout -- index.html

#* Git would not happily restore index.html! Instead, you'd get an error like this example:
# error: pathspec 'index.html' did not match any file(s) known to git.

#* To recover index.html, we would have to use a different technique: git reset. You can use git reset to unstage changes.
#* You could recover index.html by using these two commands:
git reset HEAD index.html
git checkout -- index.html
#* Here, git reset unstages the file deletion from Git. This command brings the file back to the index, but the file is still deleted on disk. You can then restore it to the disk from the index by using git checkout.

#* Here's another "Aha!" moment for new Git users. Many VCSes make files read-only to ensure that only one person at a time can make changes; users use an unrelated checkout command to get a writable version of the file. They also use checkin for an operation similar to what Git does with a combination of add, commit, and push. This fact occasionally causes confusion when people begin to use Git.




###? Revert a commit: git revert
#* The last important command to know for fixing mistakes with Git is git revert. git checkout works only in situations where the changes to undo are in the index. After you've committed changes, you need to use a different strategy to undo them. In this case, we can use git revert to revert our previous commit. It works by making another commit that cancels out the first commit.

#* We can use git revert HEAD to make a commit that's the exact opposite of our last commit, undoing the previous commit while leaving all history intact. The HEAD part of the command just tells Git that we want to "undo" only the last commit.

#* As an aside, you can also remove the most recent commit by using the git reset command:
git reset --hard HEAD^

#* Git offers several types of resets. The default is --mixed, which resets the index but not the working tree; it also moves HEAD, if you specify a different commit. The --soft option moves HEAD only, and it leaves both the index and the working tree unchanged. This option leaves all your changes as "changes to be committed", as git status would put it. A --hard reset changes both the index and the working tree to match the specified commit; any changes that you made to tracked files are discarded.