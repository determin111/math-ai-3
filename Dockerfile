# 替换原来的 FROM my-math-env
FROM python:3.9-slim

WORKDIR /app

# 配置国内镜像源，加速依赖安装
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装项目依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["python", "app.py"]
