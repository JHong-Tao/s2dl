# jekyll+GitHub搭建个人网站

## 1. 核心问题

用jekyll搭建个人完整最核心的问题，就是Ruby环境的搭建

## 2. Ruby环境搭建

### 1. 下载Ruby

官网地址：https://rubyinstaller.org/downloads/

![image-20210611192823780](https://gitee.com/jhongtao/mdpic/raw/master/mdimg/20210611192833.png)

ruby下载，这个非常的坑爹，这里红色加粗的版本是系统推荐的，我也建议就下载这个版本，一定要下载带devkit的安装包，不然后面超级麻烦，要单独安装各种工具，还容易出错，导致没办法安装jekyll

### 2. 安装Ruby

下载好了根据安装向导安装，我默认是安装到：C:\Ruby27-x64 这个位置的，这里非常坑爹，我一开始修改了安装位置到D盘我习惯软件都装到D盘，结果可嗯是我路径有问题的原因，转载D盘Ruby是没问题，可以正常使用，但是就是安装jekyll不成功，永远报错，说是gem安装扩展错误。反复试了好多次都不行，我都要放弃选择用其它框架了，最后一次我直接默认路径重装，很神奇居然成功了。

Ruby安装好了会提示安装MSYS2：

为了不必要的麻烦，我直接选择了3

![image-20210611193544373](https://gitee.com/jhongtao/mdpic/raw/master/mdimg/20210611193545.png)

等待结束后，就可以测试ruby和gem是否安装成功

输入命令测试：

```bash
ruby -v
gem -v
```

如果安装成功的话会出现对应的版本号

![image-20210611193834463](https://gitee.com/jhongtao/mdpic/raw/master/mdimg/20210611193836.png)

### 3. 安装bundler

输入安装命令

```bash
gem install bundler
```

![image-20210611193952011](https://gitee.com/jhongtao/mdpic/raw/master/mdimg/20210611193953.png)

通过如下命令查看版本，验证bundler是否安装成功

```bash
bundle -v
```

### 4. 安装Jekyll

通过如下命令安装：

```bash
gem install jekyll
```

![image-20210611194233309](https://gitee.com/jhongtao/mdpic/raw/master/mdimg/20210611194235.png)

## 3. Jekyll 主题选择

我选的主题：

Chirpy:http://jekyllthemes.org/themes/jekyll-theme-chirpy/

