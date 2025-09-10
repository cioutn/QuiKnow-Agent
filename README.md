# QuiKnow-Agent

## 简介
QuiKnow-agent是利用QuiKnow-MCP工具实现针对存储文档或者数据的智能问答和分析工具。

## 安装
```bash
git clone https://github.com/cioutn/QuiKnow-Agent.git
cd QuiKnow-Agent
# 复制.env.example，创建.env
conda create -n know-agent python=3.11 -y
conda activate know-agent
pip install -r requirements.txt
```

## 使用
```bash
python main.py check              检验api key是否正常连通
python main.py build [--path P]   启动文档/数据构建任务（调用 start_document_build）
python main.py status --job ID    查询构建任务状态（调用 get_job_status）
python main.py tree               触发目录聚类建树（调用 directory_tree_builder）
python main.py ask                进入交互问答，放入ask.md
python main.py report <your_task> 生成报告，放入report.md
```

## feature and thought
- report模式和ask模式的区别
```
ask模式是agent依据问题在目录树、文件树中寻找到相关文本并作为上下文进行单次回答的模式。
report模式先通过调用一次ask模式来了解大致的问题相关的信息，构建出多个子问题，进行文档分析或者数据SQL分析各自得到结果。最终，由一个agent来总结所有的子问题。
之前尝试过更加复杂的子问题分解，子问题之间有着先后关系，像一个图一样，但是LLM在过于细粒度的情况下反而更加容易单个节点出错而整体崩溃无效，而且重复回答，冗余水字情况存在。所以最终采用目前的中粒度的分解方式。
```

