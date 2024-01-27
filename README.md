改进前辈论文实现的多源域适应模型。

首先是按照一些依赖库

```python
pip install -r pmRequest.txt


```

然后按照apex

```shell
tar xvf apex.tar.gz
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
```

框架版本

```
python 3.8
cuda 11.3
pytorch 1.11
```

运行方式

```shell
chmod +x dist_train.sh
./dist_train.sh
```

