# git 配置和常用命令

# 1. git的逻辑

## 1.1 git的逻辑示意图

![img](https://gitee.com/jhongtao/mdpic/raw/master/mdimg/20210610125248.png)

## 1.2 git创建远程仓库和更新远程仓库逻辑

建议首先在远程仓库GitHub上创建仓库，然后clone到本地文件夹，然后pull,更新文件后add,add之后commit，commit之后push同步到远程仓库。

# 2. git命令

## 2.1 我常用的命令

```bash
git remote -v 		   
git clone [url]
git pull --all
get add .
get add [dir]
get add --all
get -am '提示信息'
git p --all
git status
```



## 2.2 常用命令

```bash
# 添加指定目录到暂存区，包括子目录
git add [dir]
 
# 添加当前目录的所有文件到暂存区
git add .

# 添加所有变化到站存区
git add --all

# 提交暂存区到仓库区
git commit -m '提示信息' 

# 提交工作区自上次commit之后的变化，直接到仓库区
git commit -a

# 显示有变更的文件
git status

# 下载远程仓库的所有变动
git fetch [remote]
 
# 显示所有远程仓库
git remote -v

 
# 下载远程仓库的所有变动
git fetch [remote]
 
# 显示所有远程仓库
git remote -v
 
# 显示某个远程仓库的信息
git remote show [remote]
 
# 增加一个新的远程仓库，并命名
git remote add [shortname] [url]
 
# 取回远程仓库的变化，并与本地分支合并
git pull [remote] [branch]
 
# 上传本地指定分支到远程仓库
git push [remote] [branch]
 
# 强行推送当前分支到远程仓库，即使有冲突
git push [remote] --force
 
# 推送所有分支到远程仓库
git push [remote] --all
```

# 3. git配置

## 3.1 git全局配置

首先确认已安装Git，可以通过 `git –version` 命令可以查看当前安装的版本。

可以通过命令 `git clone https://github.com/git/git` 进行更新

Git共有三个级别的config文件，分别是`system、global和local`。

优先级：local>global>system

在当前环境中，分别对应：

```bash
%GitPath%\mingw64\etc\gitconfig文件
$home.gitconfig文件
%RepoPath%.git\config文件
```

其中`%GitPath%`为Git的安装路径，`%RepoPath%`为某仓库的本地路径。

所以 system 配置整个系统只有一个，global 配置每个账户只有一个，而 local 配置和git仓库的数目相同，并且只有在仓库目录才能看到该配置。

大致`思路`，**建立两个密钥，不同账号配置不同的密钥，不同仓库配置不同密钥。**

```bash
git config --global user.name "你的名字"
git config --global user.email  "你的邮箱"
```

查看全局配置

```bash
git config --global --list
```

删除全局配置

```bash
git config --global --unset user.name "你的名字"
git config --global --unset user.email "你的邮箱"
```



##  3.2 生成新的 SSH keys

1. 打开当前用户下面建一个.ssh的文件夹，我的路径：C:\Users\Administrator\\.ssh
2. 在当前文件夹下右击打开gitbash,输入命令连续3次回车

```bash
ssh-keygen -t rsa -f ~/.ssh/id_rsa.github -C "xx@qq.com"
ssh-keygen -t rsa -f ~/.ssh/id_rsa.gitlab -C "xx@qq.com"
ssh-keygen -t rsa -f ~/.ssh/id_rsa.gitee -C "xx@qq.com"
```

当前目录下面成功的创建了ssh私钥和公钥

![image-20210610130202600](https://gitee.com/jhongtao/mdpic/raw/master/mdimg/20210610130204.png)

## 3.3 添加识别 SSH keys 新的私钥

```bash
ssh-agent bash
ssh-add ~/.ssh/id_rsa.github
ssh-add ~/.ssh/id_rsa.gitlab
ssh-add ~/.ssh/id_rsa.gitee
```

## 3.4 多账号必须配置 config 文件(重点)

```bash
#github
Host github.com
HostName github.com
PreferredAuthentications publickey
IdentityFile ~/.ssh/id_rsa.github

#gitlab
Host gitlab.com
HostName gitlab.com
PreferredAuthentications publickey
IdentityFile ~/.ssh/id_rsa.gitlab

#gitee
Host gitee.com
HostName gitee.com
PreferredAuthentications publickey
IdentityFile ~/.ssh/id_rsa.gitee
```

## 3.5 在 github 和 gitlab 网站添加 ssh

github

![img](https://gitee.com/jhongtao/mdpic/raw/master/mdimg/20210610130918.webp)

gitlab

![img](https://gitee.com/jhongtao/mdpic/raw/master/mdimg/20210610130947.webp)

gitee

![img](https://gitee.com/jhongtao/mdpic/raw/master/mdimg/20210610131000.webp)

## 3.6 测试是否连接成功

```bash
ssh -T git@github.com
ssh -T git@gitlab.com
ssh -T git@gitee.com
```

测试成功结果：

```bash
GitHub -> successfully
GitLab -> Welcome to GitLab
Gitee -> successfully
```

