## Feature

### Main usage
将tfrecord文件放入data_input目录并运行：
>python3 predict.py

输出结果分别存储在
`./model_output/processed_files/test_"$model"_processed_files.json`
`./model_output/processed_files/test_"$model"_classify_result.json`
文件中。

### Train new models
将/datasets/parameters.txt文件中的内容改为分类的总数。

将数据按分类放入`data_raw`文件夹下并将数据转为TFRecord格式：
>python3 convert.py

执行本命令进行训练：
>python3 train.py

导出计算图
>python3 export_inference_graph.py

模型名称为train文件夹下最新的模型
>python3 freeze_graph.py --input_checkpoint=./train/"$model.ckpt-xxxx"

将训练好的模型放入model文件夹下，并在predict.py中第114行修改model名称即可使用。
