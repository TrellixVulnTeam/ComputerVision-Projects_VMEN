# 编译proto文件
# protoc object_detection/protos/*.proto --python_out=.

# 将Slim加入PYTHONPATH
# export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# 完成安装测试
# python3 object_detection/builders/model_builder_test.py