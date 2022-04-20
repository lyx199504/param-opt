# param-opt
本项目是一个机器学习和深度学习的训练工具。本训练工具基于sklearn和pytorch，不仅支持常规训练、交叉验证训练，还支持贝叶斯搜索参数，并可随时自动保存训练模型和日志。<br>
This project is a training tool for machine learning and deep learning. Based on sklearn and pytorch, the training tool not only provides regular training, cross-validation training, but also provides Bayesian search parameters, and can automatically save training models and logs at any time.

## 目录 Table of Contents

- [项目结构 Project structure](#project)
  - [项目文件 Project Files](#project-file)
  - [附加文件 Additional Files](#additional-file)
- [使用方法 Getting Started](#get-start)
  - [安装方法 Installation](#install)
  - [训练步骤 Training Steps](#train-step)
    - [常规训练 Regular Training](#train)
    - [交叉验证训练 Cross-validation Training](#cv-train)
    - [贝叶斯搜索训练 Bayesian Search Training](#bys-train)
- [项目声明 Project Statement](#statement)
- [友情链接 Related Links](#links)
- [许可证 License](#license)

<h2 id="project">项目结构 Project structure</h2>

<h3 id="project-file">项目文件 Project Files</h3>

├─ optUtils (工具目录 Tools catalog) <br>
&emsp;├─ \_\_init\_\_.py (读写文件模块 Reading and writing files module) <br>
&emsp;├─ dataUtil.py (数据模块 Data module) <br>
&emsp;├─ logUtil.py (日志模块 Log module) <br>
&emsp;├─ metricsUtil.py (评价指标模块 Evaluation metrics module) <br>
&emsp;├─ modelUtil.py (模型模块 Model module) <br>
&emsp;├─ pytorchModel.py (深度学习模型 Deep learning model) <br>
&emsp;├─ trainUtil.py (训练模块 Training module) <br>
├─ param.yaml (配置文件 Configuration file) <br>
├─ requirements.txt (环境依赖 Dependency package) <br>

<h3 id="additional-file">附加文件 Additional Files</h3>

├─ example_dl_model.py (深度学习模型样例 Deep learning model examples) <br>
├─ example_train.py (常规训练样例 Regular training examples) <br>
├─ example_train_cv.py (交叉验证训练样例 Cross-validation training examples) <br>
├─ example_train_bys.py (贝叶斯搜索样例 Bayesian search examples) <br>

<h2 id="get-start">使用方法 Getting Started</h2>

<h3 id="install">安装方法 Installation</h3>

首先，拉取本项目到本地。<br>
First, pull the project to the local.

    $ git clone git@github.com:lyx199504/param-opt.git
接着，进入到项目中并安装本项目的依赖。但要注意，pytorch可能需要采取其他方式安装，安装完毕pytorch后可直接用如下代码安装其他依赖。<br>
Next, enter the project and install the dependencies of the project. However, it should be noted that pytorch may need to be installed in other ways. After installing pytorch, you can directly install other dependencies with the following code.

    $ cd param-opt/
    $ pip install -r requirements.txt
最后，执行任意一个训练样例文件，将会生成日志文件夹“log”和训练模型文件夹“model”。<br>
Finally, executing any training sample file will generate a log folder "log" and a training model folder "model".

<h3 id="train-step">训练步骤 Training Steps</h3>

使用本工具进行训练，需要有初步的机器学习或深度学习的经验。以下教程只能大概地提及训练基本步骤，如果要更深入地使用并修改本工具，则需要进一步阅读optUtils文件夹中的代码。<br>
Training with this tool requires preliminary machine learning or deep learning experience. The following tutorial can only briefly mention the basic steps of training, if you want to use and modify the tool in more depth, you need to read the code in the optUtils folder further.

<h4 id="train">常规训练 Regular Training</h4>

参考example_train.py。<br>
Refer to example_train.py.

机器学习常规训练：<br>
Machine learning regular training:<br>
步骤一：自行将待训练的数据封装为numpy.ndarray类型，同时可采用dataUtil.py中的分层打乱数据函数将数据打乱，并将数据切分为训练集和测试集；<br>
Step 1: Encapsulate the data to be trained as numpy.ndarray type, and use the layered data shuffling function in dataUtil.py to shuffle the data, and divide the data into training sets and test sets;<br>
步骤二：将切分好的数据、模型名称、模型参数和评价指标列表填入trainUtil.py中的ml_train函数，即可训练数据。<br>
Step 2: Fill in the segmented data, model name, model parameters and evaluation metrics list into the ml_train function in trainUtil.py to train the data.
> 其中模型名称可在modelUtil.py中的__model_dict字典查阅，评价指标可直接采用sklearn中的评价指标，或在metricsUtil.py中查阅。若要使用的模型不在字典中，可自行构建模型并填入ml_train函数中的model参数；若需要创建新的评价指标，可自行添加。<br>
> The model name can be viewed in the __model_dict dictionary in modelUtil.py, and the evaluation metrics can be directly used in sklearn, or in metricsUtil.py. If the model to be used is not in the dictionary, you can build the model yourself and fill in the model parameter in the ml_train function; if you need to create a new evaluation metric, you can add it yourself.

深度学习常规训练：<br>
Deep learning regular training:<br>
步骤一：与机器学习常规训练相同；<br>
Step 1: The same as the regular training of machine learning;<br>
步骤二：查找pytorchModel.py中的模型或自行构造模型（参考example_dl_model.py），然后填入超参数、训练数据和评价指标即可训练数据。有必要提及的一些功能如下：<br>
Step 2: Find the model in pytorchModel.py or construct your own model (refer to example_dl_model.py), and then fill in the hyperparameters, training data and evaluation indicators to train the data. Some features that are worth mentioning are as follows:<br>
    
    model.param_search = False 
    # 参数搜索开关，不使用参数搜索时需要关闭 
    # Parameter search switch, it needs to be turned off when parameter search is not used
    model.only_save_last_epoch = True
    # 若关闭则每个epoch会生成一行日志和一个训练模型，开启则只生成最后一个epoch的日志和训练模型 
    # If it is turned off, each epoch will generate a log and a training model. If it is turned on, only the log and training model of the last epoch will be generated.
    model.save_model = True
    # 保存训练模型开关，开启则会保存训练模型
    # Save the training model switch, if enabled, the training model will be saved.
    model.device = 'cuda'
    # device设置为“cuda”，则启用GPU训练模型，不设置则默认采用CPU训练
    # device is set to "cuda", the GPU training model is enabled, if not set, the CPU training is used by default.

<h4 id="cv-train">交叉验证训练 Cross-validation Training</h4>

参考example_train_cv.py。<br>
Refer to example_train_cv.py.

交叉验证训练不区分机器学习和深度学习训练，换言之，这两种训练可以使用同一个流程。<br>
Cross-validation training does not differentiate between machine learning and deep learning training, in other words, the same process can be used for both kinds of training.

交叉验证训练步骤与常规训练类似，不同之处是将ml_train函数替换为cv_train函数，当需要使用的模型不在modelUtil.py中时，则按照如下代码注册自己创建的模型：<br>
The cross-validation training steps are similar to regular training, except that the ml_train function is replaced by the cv_train function. When the model to be used is not in modelUtil.py, the model created by yourself should be registered according to the following code:

    model_registration(
        rnn_clf=RNNClassifier,
    )
> 其中rnn_clf是模型的model_name，RNNClassifier是模型构造的类名。<br>
> where rnn_clf is the model_name of the model and RNNClassifier is the class name constructed by the model.

另外，需要在param.yaml配置文件设定训练折数和进程个数，如下所示：
    
    cv_param:
      fold: 10  # 训练的折数 training fold
      workers: 1  # 进程数，即采用多少进程并发执行 The number of processes, that is, how many processes are used to execute concurrently

<h4 id="bys-train">贝叶斯搜索训练 Bayesian Search Training</h4>

example_train_bys.py。<br>
Refer to example_train_bys.py.

贝叶斯搜索在训练之前的步骤与交叉验证训练相同，同样需要注册自己构建的新模型（若有新模型的话）。<br>
The pre-training steps of Bayesian search are the same as cross-validation training, and you also need to register your own new model (if there is a new model).

在训练时，将cv_train函数替换为bayes_search_train函数，同时数据无需切分为训练集和验证集，因为在训练时会自动切分训练。<br>
During training, replace the cv_train function with the bayes_search_train function, and the data does not need to be split into training set and validation set, because the training will be automatically split during training.

另外，需要在param.yaml配置文件设定迭代次数、训练折数、进程个数、以及每个模型的参数搜索范围，代码如下：

    cv_param:
      n_iter: 10  # 迭代次数，即采用多少个参数组合训练 The number of iterations, that is, how many parameter combinations are used for training
      fold: 3  
      workers: 1 
    model:
      - [lr_clf, {
          max_iter: !!python/tuple [50, 200],
          C: !!python/tuple [0.8, 1.2, 'uniform'],
          random_state: !!python/tuple [1, 500],
      }]
> 其中，“model”下面定义每个模型，“lr_clf”是模型名称，冒号“:”的左边是参数名，右边是参数待搜索的范围，若要更熟练地使用贝叶斯搜索，可能需要进一步了解skopt包中的BayesSearchCV。<br>
> Among them, each model is defined under "model", "lr_clf" is the model name, the left side of the colon ":" is the parameter name, and the right side is the range of parameters to be searched. If you want to use Bayesian search more proficiently, you may need to further Learn about BayesSearchCV in the skopt package.

<h2 id="statement">项目声明 Project Statement</h2>

若你使用本项目用于论文的实验，你可以引用本项目，latex版本引用如下：<br>
If you use this project for the experiment of the paper, you can cite this project, the latex version is cited as follows:

    @misc{paramopt,
      author       = {Lu, Yixiang},
      title        = {param-opt: A machine learning training tool},
      year         = {2022},
      howpublished = {\url{https://github.com/lyx199504/param-opt}}
    }
word版本引用如下：<br>
The word version is quoted as follows:
    
    Y. Lu, param-opt: A machine learning training tool, https://github.com/lyx199504/param-opt (2022).

当你公开了基于本项目的代码时，你必须注明原项目作者及出处：<br>
When you disclose the code based on this project, you must indicate the original project author and source:<br>

    Author: Yixiang Lu
    Project: https://github.com/lyx199504/param-opt

<h2 id="links">友情链接 Related Links</h2>

1. [点击欺诈CAT-RFE集成学习框架](https://github.com/lyx199504/click-fraud-cat-rfe)

<h2 id="license">许可证 License</h2>

[MIT](LICENSE) (c) 2022 Yixiang Lu - 夜光
